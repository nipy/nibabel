# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Write MINC2 (HDF5) format images

This module depends only on numpy and h5py.  It writes the subset of the
MINC2 format that nibabel's MINC2 reader understands, and is the inverse of
that reader: for a written file ``fname``, ``nibabel.load(fname)`` returns
the written data, and an affine equal to the written affine up to floating
point rounding (exactly so for affines whose columns are scaled coordinate
axes).

See the `MINC2 format reference
<https://en.wikibooks.org/wiki/MINC/Reference/MINC2.0_File_Format_Reference>`_.

Layout
------
The file has a single group ``/minc-2.0`` with the attributes ``ident``,
``minc_version`` and ``history``, and the following members:

* ``dimensions/{x,y,z}space`` (and ``dimensions/time`` for 4D images):
  regularly spaced dimension variables with ``start``, ``step``,
  ``direction_cosines`` (spatial dimensions only), ``spacing='regular__'``,
  ``alignment='centre'``, ``length`` and ``units``;
* ``image/0/image``: the voxel data, with ``dimorder`` and ``valid_range``;
* ``image/0/image-min``, ``image/0/image-max``: the real value range;
* ``info``: an empty group.

Axis order
----------
The array axes are the MINC dimensions in file order, slowest varying first,
as nibabel's reader returns them.  For a 3D array, the three axes are spatial,
and column ``i`` of the affine belongs to array axis ``i``.  For a 4D array,
axis 0 is ``time`` (MINC stores time as the slowest varying dimension), and
column ``i`` of the affine belongs to array axis ``i + 1``.  Arrays with time
last, as in NIfTI, need ``np.moveaxis(data, -1, 0)`` first.

Each spatial axis is named ``xspace``, ``yspace`` or ``zspace`` after the world
axis its direction is closest to.  The direction cosines are unit vectors, and
the sign of the axis direction goes into ``step``, so that each cosine points
along the positive direction of the world axis that names it, as in libminc.

Scaling
-------
The ``image-min`` and ``image-max`` variables are written slice-wise, as
libminc does by default: one value per index of all but the two fastest
varying dimensions (one per ``zspace`` slice for a 3D ``zspace, yspace,
xspace`` image; a ``(time, slice)`` array for a 4D image).

* Floating point output (float32, float64) stores the data unscaled; the
  image-min / image-max hold the slice minima and maxima, and ``valid_range``
  the minimum and maximum of the whole volume.
* Integer output (uint8, int16, uint16, int32) has ``valid_range`` equal to
  the range of the type.  If every value is an integer within that range,
  the data are stored unscaled (image-min and image-max equal the valid
  range, so that the scale factor is exactly 1).  Otherwise each slice is
  scaled linearly from its own minimum and maximum onto the valid range and
  rounded to the nearest integer, so the error per voxel is at most half a
  quantisation step of its slice.
* Integer output with explicit ``scaling=(image_min, image_max,
  valid_range)`` uses those values as given.  Re-writing the real values
  that nibabel read from a file with that file's scaling reproduces the
  stored integers, and hence the real values, exactly.
"""

import getpass
import itertools
import os
import socket
import time
import warnings

import numpy as np

#: The libminc release whose file layout this writer follows.  It is
#: written to the ``minc_version`` attribute, as libminc writes its version.
MINC_VERSION = '2.4.01'

#: Data types that can be written.
SUPPORTED_DTYPES = tuple(
    np.dtype(t) for t in (np.uint8, np.int16, np.uint16, np.int32, np.float32, np.float64)
)

_SPATIAL_NAMES = ('xspace', 'yspace', 'zspace')
_COMMENTS = {
    'xspace': 'X increases from patient left to right',
    'yspace': 'Y increases from patient posterior to anterior',
    'zspace': 'Z increases from patient inferior to superior',
}
_STD_VARID = 'MINC standard variable'
_STD_VERSION = 'MINC Version    1.0'

_ident_count = itertools.count(1)


class Minc2WriteError(Exception):
    """Error when writing a MINC2 file"""


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
        raise Minc2WriteError(f'Affine must have shape (4, 4), not {affine.shape}')
    if not np.all(np.isfinite(affine)):
        raise Minc2WriteError('Affine must be finite')
    rzs = affine[:3, :3]
    norms = np.sqrt(np.sum(rzs**2, axis=0))
    if np.any(norms == 0):
        raise Minc2WriteError('Affine has a zero voxel size')
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
        raise Minc2WriteError('Affine is singular')
    names = tuple(_SPATIAL_NAMES[p] for p in perm)
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


def _string_attr(obj, name, value):
    """Set ASCII string attribute as a null-terminated fixed length string"""
    import h5py

    value = value.encode('ascii')
    tid = h5py.h5t.C_S1.copy()
    tid.set_size(len(value) + 1)
    tid.set_strpad(h5py.h5t.STR_NULLTERM)
    obj.attrs.create(name, np.bytes_(value), dtype=h5py.Datatype(tid))


def _std_attrs(obj, vartype):
    _string_attr(obj, 'varid', _STD_VARID)
    _string_attr(obj, 'vartype', vartype)
    _string_attr(obj, 'version', _STD_VERSION)


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
            raise Minc2WriteError(f'Cannot write non-finite values as {dtype}')
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


def _write_dimension(group, name, length, start, step, cosine=None, units='mm'):
    var = group.create_dataset(name, shape=(), dtype='i4')
    _string_attr(var, 'alignment', 'centre')
    if name in _COMMENTS:
        _string_attr(var, 'comments', _COMMENTS[name])
    if cosine is not None:
        var.attrs.create('direction_cosines', np.asarray(cosine, dtype=np.float64))
    var.attrs.create('length', np.uint32(length))
    if cosine is not None:
        _string_attr(var, 'spacetype', 'native____')
    _string_attr(var, 'spacing', 'regular__')
    var.attrs.create('start', np.float64(start))
    var.attrs.create('step', np.float64(step))
    _string_attr(var, 'units', units)
    _std_attrs(var, 'dimension____')


def _write_range_var(group, name, values, dim_names):
    values = np.asarray(values, dtype=np.float64)
    var = group.create_dataset(name, data=values)
    if values.ndim:
        _string_attr(var, 'dimorder', ','.join(dim_names[: values.ndim]))
    _std_attrs(var, 'var_attribute')


def write_minc2(
    data,
    affine,
    filename,
    dtype=None,
    history=None,
    scaling=None,
    time_start=0.0,
    time_step=1.0,
):
    """Write `data` and `affine` to a MINC2 file

    Parameters
    ----------
    data : array-like
        3D array (three spatial axes), or 4D array with time on axis 0 and
        the spatial axes after it.  The axes are the MINC dimensions in file
        order, slowest varying first, as nibabel's MINC reader returns them.
    affine : array-like, shape (4, 4)
        Voxel to world affine of the spatial axes.
    filename : str, os.PathLike or file-like
        Output file name, or a binary file object open for writing and
        seeking (as accepted by ``h5py.File``).
    dtype : numpy dtype specifier, optional
        Stored data type: one of uint8, int16, uint16, int32, float32 and
        float64.  Default is the data type of `data`.
    history : str, optional
        Text of the ``history`` attribute.  Default is a single libminc style
        line, ``'<ctime>>>> write_minc2 <filename>'``.  A string not ending
        in a newline gets one.
    scaling : tuple, optional
        ``(image_min, image_max, valid_range)`` for integer output, where
        `image_min` and `image_max` are scalars or arrays covering the first
        0, 1 or 2 axes of `data`, and `valid_range` is ``(vmin, vmax)``.
        Default chooses the scaling as described in the module docstring.
        Ignored for floating point output.
    time_start, time_step : float, optional
        ``start`` and ``step`` of the time dimension of a 4D image.

    Returns
    -------
    None
    """
    import h5py

    data = np.asanyarray(data)
    if data.ndim not in (3, 4):
        raise Minc2WriteError(f'Can only write 3D or 4D data, not {data.ndim}D')
    out_dtype = np.dtype(data.dtype if dtype is None else dtype).newbyteorder('=')
    if out_dtype not in SUPPORTED_DTYPES:
        allowed = ', '.join(str(t) for t in SUPPORTED_DTYPES)
        raise Minc2WriteError(f'Cannot write data type {out_dtype}; use one of {allowed}')
    names, starts, steps, cosines = affine_to_dimensions(affine)
    dim_names = list(names)
    if data.ndim == 4:
        dim_names.insert(0, 'time')

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
                raise Minc2WriteError('image-min / image-max do not match the data shape')
            if image_min.ndim > 2:
                raise Minc2WriteError('More than two scaling dimensions')
        raw = _quantise(data, out_dtype, image_min, image_max, valid_range)

    if history is None:
        target = filename if isinstance(filename, (str, os.PathLike)) else '<file object>'
        history = make_history_line(f'write_minc2 {os.fspath(target)}')
    elif not history.endswith('\n'):
        history += '\n'

    def write(h5f):
        _write_groups(
            h5f,
            raw,
            dim_names,
            (names, starts, steps, cosines),
            (image_min, image_max, valid_range),
            history,
            (time_start, time_step),
        )

    if isinstance(filename, (str, os.PathLike)):
        filename = os.fspath(filename)
    with h5py.File(filename, 'w') as h5f:
        write(h5f)


def _write_groups(h5f, raw, dim_names, geometry, ranges, history, time_dim):
    """Write the ``/minc-2.0`` group of a MINC2 file to open ``h5f``"""
    names, starts, steps, cosines = geometry
    image_min, image_max, valid_range = ranges
    time_start, time_step = time_dim
    minc = h5f.create_group('minc-2.0')
    _string_attr(minc, 'ident', make_ident())
    _string_attr(minc, 'minc_version', MINC_VERSION)
    _string_attr(minc, 'history', history)
    dims = minc.create_group('dimensions')
    if raw.ndim == 4:
        _write_dimension(dims, 'time', raw.shape[0], time_start, time_step, units='s')
    spatial_shape = raw.shape[-3:]
    for i, name in enumerate(names):
        _write_dimension(dims, name, spatial_shape[i], starts[i], steps[i], cosines[:, i])
    minc.create_group('info')
    group = minc.create_group('image').create_group('0')
    image = group.create_dataset('image', data=raw)
    _string_attr(image, 'complete', 'true_')
    _string_attr(image, 'dimorder', ','.join(dim_names))
    image.attrs.create('valid_range', np.asarray(valid_range, dtype=np.float64))
    _std_attrs(image, 'group________')
    _write_range_var(group, 'image-max', image_max, dim_names)
    _write_range_var(group, 'image-min', image_min, dim_names)
