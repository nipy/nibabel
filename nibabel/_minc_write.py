# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Format independent part of the MINC writers

:func:`build_layout` turns data, an affine and options into a
:class:`Layout`: the stored values, the dimensions with their attributes,
the scaling, and the global and image attributes.  The MINC2 writer in
:mod:`nibabel.minc2_write` stores a layout in HDF5.
"""

import getpass
import itertools
import os
import socket
import time
import uuid
import warnings
from dataclasses import dataclass, field

import numpy as np

#: The libminc release whose file layout the writers follow.  It is written
#: to the ``minc_version`` attribute, as libminc writes its version.
MINC_VERSION = '2.4.01'

#: Data types that can be written.
SUPPORTED_DTYPES = tuple(
    np.dtype(t) for t in (np.uint8, np.int16, np.uint16, np.int32, np.float32, np.float64)
)

SPATIAL_NAMES = ('xspace', 'yspace', 'zspace')
COMMENTS = {
    'xspace': 'X increases from patient left to right',
    'yspace': 'Y increases from patient posterior to anterior',
    'zspace': 'Z increases from patient inferior to superior',
}
STD_VARID = 'MINC standard variable'
STD_VERSION = 'MINC Version    1.0'

#: Attributes the writers work out from the data and the affine, and never
#: copy from metadata
GEOMETRY_ATTRS = frozenset(('start', 'step', 'direction_cosines', 'length', 'spacing'))
#: Attributes that describe the structure of a file, written by the writers
STRUCTURE_ATTRS = frozenset(
    (
        'dimorder',
        'parent',
        'children',
        'image-min',
        'image-max',
        'signtype',
        '_FillValue',
        'complete',
        'valid_range',
        'valid_min',
        'valid_max',
    )
)
#: Attributes that describe a dimension, kept from metadata
DESCRIPTION_ATTRS = frozenset(('class', 'width', 'alignment'))
#: Global attributes that belong to the file being written
FILE_ATTRS = frozenset(('ident', 'minc_version', 'history'))

_ident_count = itertools.count(1)


class MincWriteError(Exception):
    """Error when writing a MINC file"""


def affine_to_dimensions(affine):
    """Convert a voxel to world affine to MINC dimension attributes

    Parameters
    ----------
    affine : array-like, shape (4, 4)
        Voxel to world affine.  Column ``i`` of ``affine[:3, :3]`` is the
        world displacement of one voxel along spatial array axis ``i``.

    Returns
    -------
    names : tuple of str
        MINC dimension name (``'xspace'``, ``'yspace'`` or ``'zspace'``) of
        each spatial array axis.
    starts : ndarray, shape (3,)
        World coordinate of the first voxel centre along each axis, in the
        direction cosine frame (MINC ``start``).
    steps : ndarray, shape (3,)
        Signed voxel size along each axis (MINC ``step``).
    cosines : ndarray, shape (3, 3)
        Unit direction cosine of each axis, in the columns.

    Notes
    -----
    nibabel's reader forms the affine as ``cosines * steps`` with translation
    ``cosines @ starts``, so this is its inverse.  Each axis is named after
    the world axis its cosine is closest to, choosing among the six
    assignments the one with the largest sum of absolute cosine components;
    the cosine is then flipped, and the step negated, so that this component
    is positive.
    """
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4):
        raise MincWriteError(f'Affine must have shape (4, 4), not {affine.shape}')
    if not np.all(np.isfinite(affine)):
        raise MincWriteError('Affine must be finite')
    rzs = affine[:3, :3]
    norms = np.sqrt(np.sum(rzs**2, axis=0))
    if np.any(norms == 0):
        raise MincWriteError('Affine has a zero voxel size')
    cosines = rzs / norms
    perm = max(
        itertools.permutations(range(3)),
        key=lambda p: sum(abs(cosines[p[i], i]) for i in range(3)),
    )
    signs = np.array([1.0 if cosines[perm[i], i] >= 0 else -1.0 for i in range(3)])
    cosines = cosines * signs
    steps = norms * signs
    try:
        starts = np.linalg.solve(cosines, affine[:3, 3])
    except np.linalg.LinAlgError:
        raise MincWriteError('Affine is singular')
    names = tuple(SPATIAL_NAMES[p] for p in perm)
    return names, starts, steps, cosines


def make_ident():
    """Return an ``ident`` string in the form libminc writes

    The form is ``user:host:YYYY.MM.DD.HH.MM.SS:pid:count``.
    """
    try:
        user = getpass.getuser()
    except Exception:
        user = 'unknown'
    stamp = time.strftime('%Y.%m.%d.%H.%M.%S', time.localtime())
    return f'{user}:{socket.gethostname()}:{stamp}:{os.getpid()}:{next(_ident_count)}'


def make_history_line(command):
    """Return a history line in the form libminc writes

    Parameters
    ----------
    command : str
        Command that made the file.

    Returns
    -------
    line : str
        ``'<ctime>>>> <command>\\n'``, as libminc's ``miappend_history``.
    """
    return f'{time.ctime()}>>> {command}\n'


def _slice_shape(shape):
    """Shape of slice-wise scaling arrays: all but the two fastest axes"""
    return shape[:-2]


def _slice_reduce(func, data):
    """Reduce `data` over its two fastest axes with `func`"""
    return func(data, axis=(-2, -1))


def _float_ranges(raw):
    """Return image-min, image-max and valid range for floating point `raw`

    Non-finite values are left out; a slice without finite values has range
    0 to 0.
    """
    finite = np.isfinite(raw)
    if not finite.any():
        sshape = _slice_shape(raw.shape)
        return np.zeros(sshape), np.zeros(sshape), (0.0, 0.0)
    masked = np.where(finite, raw, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # All-NaN slices
        image_min = np.nan_to_num(_slice_reduce(np.nanmin, masked))
        image_max = np.nan_to_num(_slice_reduce(np.nanmax, masked))
    valid_range = (float(raw[finite].min()), float(raw[finite].max()))
    return image_min.astype(np.float64), image_max.astype(np.float64), valid_range


def _choose_scaling(data, dtype):
    """Return image-min, image-max and valid range for integer `dtype`"""
    info = np.iinfo(dtype)
    valid_range = (float(info.min), float(info.max))
    sshape = _slice_shape(data.shape)
    if data.size == 0:
        return np.zeros(sshape), np.ones(sshape), valid_range
    if np.issubdtype(data.dtype, np.integer):
        integral = True
    else:
        if not np.all(np.isfinite(data)):
            raise MincWriteError(f'Cannot write non-finite values as {dtype}')
        integral = np.array_equal(data, np.rint(data))
    if integral and data.min() >= info.min and data.max() <= info.max:
        return np.full(sshape, valid_range[0]), np.full(sshape, valid_range[1]), valid_range
    image_min = _slice_reduce(np.min, data).astype(np.float64)
    image_max = _slice_reduce(np.max, data).astype(np.float64)
    return image_min, image_max, valid_range


def _quantise(data, dtype, image_min, image_max, valid_range):
    """Map real `data` to stored integers, inverting the reader's scaling

    The reader computes ``real = stored * slope + inter`` with ``slope =
    (imax - imin) / (vmax - vmin)`` and ``inter = imin - vmin * slope``; we
    compute the same `slope` and `inter` and round.
    """
    vmin, vmax = (float(v) for v in valid_range)
    image_min = np.asarray(image_min, dtype=np.float64)
    image_max = np.asarray(image_max, dtype=np.float64)
    bshape = image_min.shape + (1,) * (data.ndim - image_min.ndim)
    imin = image_min.reshape(bshape)
    imax = image_max.reshape(bshape)
    slope = (imax - imin) / (vmax - vmin)
    inter = imin - vmin * slope
    real = np.asarray(data, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = np.where(slope != 0, (real - inter) / slope, vmin)
    raw = np.clip(np.rint(raw), vmin, vmax)
    return raw.astype(dtype)


def std_attrs(vartype):
    """The ``varid``, ``vartype`` and ``version`` attributes of libminc"""
    return {'varid': STD_VARID, 'vartype': vartype, 'version': STD_VERSION}


@dataclass
class Dimension:
    """A MINC dimension to write

    Attributes
    ----------
    name : str
        Dimension name.
    length : int
        Number of samples.
    attrs : dict
        Attributes of the dimension variable, in the order to write them.
    """

    name: str
    length: int
    attrs: dict = field(default_factory=dict)


def _copy_attrs(attrs, source, skip):
    """Add the attributes of `source` to `attrs`, except those in `skip`"""
    for name, value in source.items():
        if name not in skip:
            attrs[name] = value


@dataclass
class Layout:
    """Everything a MINC writer stores, independent of the file format

    Attributes
    ----------
    raw : ndarray
        Stored values, axes in the order of `dims`.
    dims : list of Dimension
        Image dimensions, slowest varying first.
    global_attrs : dict
        Global attributes, in the order to write them.
    image_attrs : dict
        Attributes of the image variable, other than its dimensions.
    image_min, image_max : ndarray
        Real value range, over the first ``image_min.ndim`` dimensions.
    info : list of MincVariable
        Info variables, from metadata.
    """

    raw: np.ndarray
    dims: list
    global_attrs: dict
    image_attrs: dict
    image_min: np.ndarray
    image_max: np.ndarray
    info: list = field(default_factory=list)

    @property
    def dimorder(self):
        return [dim.name for dim in self.dims]


def check_dtype(data, dtype):
    """Return the stored data type for `data` and requested `dtype`"""
    out_dtype = np.dtype(data.dtype if dtype is None else dtype).newbyteorder('=')
    if out_dtype not in SUPPORTED_DTYPES:
        allowed = ', '.join(str(t) for t in SUPPORTED_DTYPES)
        raise MincWriteError(f'Cannot write data type {out_dtype}; use one of {allowed}')
    return out_dtype


def build_layout(
    data,
    affine,
    *,
    dtype=None,
    history=None,
    scaling=None,
    time_start=0.0,
    time_step=1.0,
    metadata=None,
    command='write_minc',
):
    """Work out what to store for `data` and `affine`

    Parameters are as for :func:`nibabel.minc2_write.write_minc2`;
    `command` names the writer in the default history line.

    With `metadata`, a :class:`~nibabel.minc_meta.MincMeta`, the info
    variables, and the global, image and dimension attributes are copied,
    except for those the writer works out from the data and the affine: the
    geometry (``start``, ``step``, ``direction_cosines``, ``length``,
    ``spacing``), the scaling (``valid_range`` and the image-min /
    image-max variables) and the attributes that describe the file
    structure (``dimorder``, ``signtype`` and the MINC1 links between
    variables).

    Returns
    -------
    layout : Layout
    """
    data = np.asanyarray(data)
    if data.ndim not in (3, 4):
        raise MincWriteError(f'Can only write 3D or 4D data, not {data.ndim}D')
    out_dtype = check_dtype(data, dtype)
    names, starts, steps, cosines = affine_to_dimensions(affine)

    if out_dtype.kind == 'f':
        raw = data.astype(out_dtype)
        image_min, image_max, valid_range = _float_ranges(raw)
    else:
        if scaling is None:
            image_min, image_max, valid_range = _choose_scaling(data, out_dtype)
        else:
            image_min, image_max, valid_range = scaling
            image_min = np.asarray(image_min, dtype=np.float64)
            image_max = np.asarray(image_max, dtype=np.float64)
            if (
                image_min.shape != image_max.shape
                or image_min.shape != data.shape[: image_min.ndim]
            ):
                raise MincWriteError('image-min / image-max do not match the data shape')
            if image_min.ndim > 2:
                raise MincWriteError('More than two scaling dimensions')
        raw = _quantise(data, out_dtype, image_min, image_max, valid_range)

    if history is None:
        history = make_history_line(command)
    elif not history.endswith('\n'):
        history += '\n'

    dims = []
    if raw.ndim == 4:
        dims.append(
            Dimension(
                'time',
                raw.shape[0],
                {
                    'alignment': 'centre',
                    'length': np.uint32(raw.shape[0]),
                    'spacing': 'regular__',
                    'start': float(time_start),
                    'step': float(time_step),
                    'units': 's',
                    **std_attrs('dimension____'),
                },
            )
        )
    spatial_shape = raw.shape[-3:]
    for i, name in enumerate(names):
        dims.append(
            Dimension(
                name,
                spatial_shape[i],
                {
                    'alignment': 'centre',
                    'comments': COMMENTS[name],
                    'direction_cosines': cosines[:, i],
                    'length': np.uint32(spatial_shape[i]),
                    'spacetype': 'native____',
                    'spacing': 'regular__',
                    'start': float(starts[i]),
                    'step': float(steps[i]),
                    'units': 'mm',
                    **std_attrs('dimension____'),
                },
            )
        )
    global_attrs = {'ident': make_ident(), 'minc_version': MINC_VERSION, 'history': history}
    image_attrs = {
        'complete': 'true_',
        'valid_range': np.asarray(valid_range, dtype=np.float64),
        **std_attrs('group________'),
    }
    info = []
    if metadata is not None:
        _copy_attrs(global_attrs, metadata.global_attrs, FILE_ATTRS)
        _copy_attrs(image_attrs, metadata.image_attrs, STRUCTURE_ATTRS)
        skip = GEOMETRY_ATTRS | STRUCTURE_ATTRS | DESCRIPTION_ATTRS
        for dim in dims:
            source = metadata.variables.get(dim.name)
            if source is not None and source.group == 'dimensions':
                _copy_attrs(dim.attrs, source.attrs, skip)
        info = list(metadata.info.values())
    return Layout(
        raw=raw,
        dims=dims,
        global_attrs=global_attrs,
        image_attrs=image_attrs,
        image_min=np.asarray(image_min, dtype=np.float64),
        image_max=np.asarray(image_max, dtype=np.float64),
        info=info,
    )


def write_replacing(filename, write):
    """Call ``write(path)`` on a new file and rename it to `filename`

    The file is written beside its final name and renamed into place, so
    that a file that is open for reading (for example by the image being
    saved) is not truncated.  Creating the file with mode 0o666 applies the
    umask.
    """
    filename = os.fspath(filename)
    dirname, basename = os.path.split(os.path.abspath(filename))
    tmp_name = os.path.join(dirname, f'.{basename}.{uuid.uuid4().hex}.tmp')
    os.close(os.open(tmp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
    try:
        write(tmp_name)
        os.replace(tmp_name, filename)
    except BaseException:
        os.unlink(tmp_name)
        raise
