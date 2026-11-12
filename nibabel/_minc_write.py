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
:mod:`nibabel.minc2_write` stores a layout in HDF5, and the MINC1 writer in
:mod:`nibabel.minc1_write` in NetCDF.
"""

from __future__ import annotations

import getpass
import itertools
import os
import socket
import time
import uuid
import warnings
from dataclasses import dataclass, field

import numpy as np

from .minc_meta import WIDTH_SUFFIX, WORLD_AXES, MincVariable

VECTOR_NAME = 'vector_dimension'

#: The libminc release whose file layout the writers follow.  It is written
#: to the ``minc_version`` attribute, as libminc writes its version.
MINC_VERSION = '2.4.01'

#: Data types that can be written.
SUPPORTED_DTYPES = tuple(
    np.dtype(t)
    for t in (
        np.uint8,
        np.int8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    )
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
GEOMETRY_ATTRS = ('spacing', 'step', 'start', 'direction_cosines', 'length')
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
#: Global attributes that belong to the file being written
FILE_ATTRS = frozenset(('ident', 'minc_version', 'history'))

#: Value of MINC variables whose data are not used, as libminc 2.4 writes
MISSING_INT = 0

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


def _slice_reduce(func, data, nscale):
    """Reduce `data` over all but its first `nscale` axes with `func`"""
    return func(data, axis=tuple(range(nscale, data.ndim)))


def _float_ranges(raw, nscale):
    """Return image-min, image-max and valid range for floating point `raw`

    The image-min and image-max vary over the first `nscale` axes.
    Non-finite values are left out; a slice without finite values has range
    0 to 0.
    """
    finite = np.isfinite(raw)
    if not finite.any():
        sshape = raw.shape[:nscale]
        return np.zeros(sshape), np.zeros(sshape), (0.0, 0.0)
    masked = np.where(finite, raw, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # All-NaN slices
        image_min = np.nan_to_num(_slice_reduce(np.nanmin, masked, nscale))
        image_max = np.nan_to_num(_slice_reduce(np.nanmax, masked, nscale))
    valid_range = (float(raw[finite].min()), float(raw[finite].max()))
    return image_min.astype(np.float64), image_max.astype(np.float64), valid_range


def _choose_scaling(data, dtype, nscale):
    """Return image-min, image-max and valid range for integer `dtype`

    The image-min and image-max vary over the first `nscale` axes.
    """
    info = np.iinfo(dtype)
    valid_range = (float(info.min), float(info.max))
    sshape = data.shape[:nscale]
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
    image_min = _slice_reduce(np.min, data, nscale).astype(np.float64)
    image_max = _slice_reduce(np.max, data, nscale).astype(np.float64)
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
    data : None or ndarray
        Data of the dimension variable: a scalar, or the coordinate of each
        sample of an irregularly spaced dimension.  None writes the default
        scalar.
    widths : None or ndarray
        Width of each sample, for the ``<name>-width`` variable, or None.
    width_attrs : dict
        Attributes of the ``<name>-width`` variable.
    width_group : str
        MINC2 group of the ``<name>-width`` variable: ``'dimensions'``, as
        libminc's MINC2 functions write it, or ``'info'``, as its MINC1
        functions do in MINC2 files.
    """

    name: str
    length: int
    attrs: dict = field(default_factory=dict)
    data: np.ndarray | None = None
    widths: np.ndarray | None = None
    width_attrs: dict = field(default_factory=dict)
    width_group: str = 'dimensions'


def _source_geometry(source, metadata):
    """Length, start, step and cosine of dimension variable `source`"""
    attrs = source.attrs
    coords = None
    if attrs.get('spacing') == 'irregular' and source.data is not None and source.data.ndim == 1:
        coords = np.asarray(source.data, dtype=np.float64)
    if 'start' in attrs:
        start = float(attrs['start'])
    else:
        start = float(coords[0]) if coords is not None and coords.size else 0.0
    if 'step' in attrs:
        step = float(attrs['step'])
    elif coords is not None and coords.size > 1:
        step = float(coords[-1] - coords[0]) / (coords.size - 1)
    else:
        step = 1.0
    cosine = attrs.get('direction_cosines')
    if cosine is None and source.name in WORLD_AXES:
        cosine = np.eye(3)[WORLD_AXES[source.name]]
    return metadata.dim_lengths.get(source.name), start, step, cosine


def _same_geometry(source, metadata, length, start, step, cosine):
    """True if dimension `source` has the given length and geometry

    For spatial dimensions, the voxel axis (cosine times step) and the
    offset along it (cosine times start) must agree; a cosine and step may
    both have their signs flipped.
    """
    src_length, src_start, src_step, src_cosine = _source_geometry(source, metadata)
    if src_length != length:
        return False
    if cosine is None:
        return True
    if src_cosine is None or np.shape(src_cosine) != (3,):
        return False
    src_cosine = np.asarray(src_cosine, dtype=np.float64)
    tol = 1e-6 * max(abs(step), 1.0)
    return np.allclose(src_cosine * src_step, cosine * step, rtol=1e-6, atol=tol) and np.allclose(
        src_cosine * src_start, cosine * start, rtol=1e-6, atol=1e-6 * max(abs(start), 1.0)
    )


def _width_variable(metadata, name):
    var = metadata.variables.get(name + WIDTH_SUFFIX)
    if var is None or var.data is None or var.data.ndim != 1:
        return None
    return var


def _is_width_of(var, dims):
    return any(var.name == dim.name + WIDTH_SUFFIX for dim in dims)


def _dimension(name, length, metadata, start=None, step=None, cosine=None):
    """Return the :class:`Dimension` to write

    Spatial dimensions (with a `cosine`) take `start` and `step` from the
    affine.  Other dimensions take them from the arguments if given, from
    `metadata` if its dimension of that name has the same length, and are
    0 and 1 otherwise.

    The dimension variable of `metadata` with the same name gives the
    other attributes.  If its geometry matches, its coordinates and sample
    widths (irregular spacing) and its exact ``start``, ``step`` and
    ``direction_cosines`` are kept as well; otherwise the dimension is
    written regularly spaced.
    """
    spatial = cosine is not None
    source = None
    if metadata is not None:
        source = metadata.variables.get(name)
        if source is not None and source.group != 'dimensions':
            source = None
    explicit = not spatial and (start is not None or step is not None)
    keep = False
    if source is not None and not explicit:
        if spatial:
            keep = _same_geometry(source, metadata, length, start, step, cosine)
        else:
            keep = metadata.dim_lengths.get(name) == length
            if keep:
                _, start, step, _ = _source_geometry(source, metadata)
    start = 0.0 if start is None else float(start)
    step = 1.0 if step is None else float(step)
    # The attributes and their order of libminc's MINC1 dimension variables
    attrs = dict(std_attrs('dimension____'))
    if name in COMMENTS:
        attrs['comments'] = COMMENTS[name]
    attrs['spacing'] = 'regular__'
    # libminc aligns time samples to their start
    attrs['alignment'] = 'start_' if name == 'time' else 'centre'
    attrs['step'] = step
    if name in COMMENTS or name == 'time':
        attrs['units'] = 'mm' if spatial else 's'
    attrs['start'] = start
    if spatial:
        attrs['direction_cosines'] = np.asarray(cosine, dtype=np.float64)
    attrs['length'] = np.uint32(length)
    dim = Dimension(name, length, attrs)
    if source is None:
        return dim
    # The attributes are those of the source, in its order, with the
    # geometry of the data and the affine
    geometry = {att: attrs[att] for att in GEOMETRY_ATTRS if att in attrs}
    if keep:
        # The geometry matches, so keep it as the source has it, including
        # attributes left to their defaults
        for att in ('start', 'step', 'direction_cosines'):
            geometry.pop(att, None)
            if att in source.attrs:
                geometry[att] = source.attrs[att]
        if source.data is not None and source.data.shape in ((), (length,)):
            dim.data = source.data
        width = _width_variable(metadata, name)
        if width is not None and width.data.shape == (length,):
            dim.widths = width.data
            _copy_attrs(dim.width_attrs, width.attrs, STRUCTURE_ATTRS | {'length'})
            dim.width_group = width.group
    if 'spacing' not in source.attrs:
        del geometry['spacing']
    elif keep and source.attrs['spacing'] == 'irregular' and dim.data is not None:
        geometry['spacing'] = 'irregular' if dim.data.ndim else 'regular__'
    dim.attrs = {}
    for att, value in source.attrs.items():
        if att in GEOMETRY_ATTRS:
            if att in geometry:
                dim.attrs[att] = geometry.pop(att)
        elif att not in STRUCTURE_ATTRS:
            dim.attrs[att] = value
    dim.attrs.update(geometry)
    return dim


def _record_dimension(record_dimension, dims, metadata):
    """Name of the record dimension to write, or None"""
    first = dims[0].name
    if record_dimension is None:
        if metadata is None or metadata.record_dimension != first:
            return None
        return first
    if record_dimension is False:
        return None
    if record_dimension is True or record_dimension == first:
        return first
    raise MincWriteError(
        f'The record dimension must be the first dimension, {first}, not {record_dimension}'
    )


def _world_names(names, metadata, dimension_names=None):
    """Dimension names of the spatial axes, ``space`` or ``frequency``

    `names` are the ``space`` names the affine gives the axes.  libminc
    maps ``xfrequency`` to the world x axis as it does ``xspace``, and so
    on.
    """
    axes = [WORLD_AXES[name] for name in names]
    if dimension_names is not None:
        dimension_names = tuple(dimension_names)
        if len(dimension_names) != 3 or any(
            WORLD_AXES.get(new) != axis for new, axis in zip(dimension_names, axes)
        ):
            raise MincWriteError(
                f'Spatial dimension names {dimension_names} do not match the world axes '
                f'{names} of the affine'
            )
        return dimension_names
    if metadata is None:
        return names
    by_axis = {WORLD_AXES[name]: name for name in metadata.dimorder if name in WORLD_AXES}
    return tuple(by_axis.get(axis, name) for name, axis in zip(names, axes))


def _vector_dimension(length, metadata):
    """The vector dimension, the fastest varying, such as RGB components

    libminc's MINC1 functions make no variable for it; its MINC2 functions
    make a dataset with only a ``length`` attribute.
    """
    dim = Dimension(VECTOR_NAME, length, {'length': np.uint32(length)})
    source = None if metadata is None else metadata.variables.get(VECTOR_NAME)
    if source is not None:
        dim.attrs = {}
        _copy_attrs(dim.attrs, source.attrs, STRUCTURE_ATTRS)
        dim.attrs['length'] = np.uint32(length)
        if source.data is not None and source.data.shape == ():
            dim.data = source.data
    return dim


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
    scaling_attrs : dict
        Attributes of the ``image-max`` and ``image-min`` variables, by name.
    info : list of MincVariable
        Info variables, from metadata.
    metadata : None or MincMeta
        The metadata the layout copies from.
    labels : None or dict
        Name of each label value, for a label volume.
    record_dimension : None or str
        Name of the record (unlimited) dimension, the first, if any.
    """

    raw: np.ndarray
    dims: list
    global_attrs: dict
    image_attrs: dict
    image_min: np.ndarray
    image_max: np.ndarray
    scaling_attrs: dict = field(default_factory=dict)
    info: list = field(default_factory=list)
    metadata: object = None
    labels: dict | None = None
    record_dimension: str | None = None

    @property
    def dimorder(self):
        return [dim.name for dim in self.dims]


def check_dtype(data, dtype, dtypes=SUPPORTED_DTYPES):
    """Return the stored data type for `data` and requested `dtype`

    Integer complex types, which MINC2 defines, are not supported, as
    libminc defines no scaling for them.
    """
    out_dtype = np.dtype(data.dtype if dtype is None else dtype).newbyteorder('=')
    if out_dtype not in dtypes:
        allowed = ', '.join(str(t) for t in dtypes)
        raise MincWriteError(f'Cannot write data type {out_dtype}; use one of {allowed}')
    return out_dtype


def build_layout(
    data,
    affine,
    *,
    dtype=None,
    dtypes=SUPPORTED_DTYPES,
    history=None,
    scaling=None,
    time_start=None,
    time_step=None,
    metadata=None,
    vector_dimension=None,
    slice_scaling=None,
    labels=None,
    dimension_names=None,
    record_dimension=None,
    command='write_minc',
):
    """Work out what to store for `data` and `affine`

    Parameters are as for :func:`nibabel.minc2_write.write_minc2`;
    `dtypes` are the data types the format can store, and `command` names
    the writer in the default history line.

    `labels` maps label values to names, for a label volume; default is the
    labels of `metadata`, if any.  The data of a label volume must be
    integers that the stored type can hold, and are stored unscaled.

    `dimension_names` names the spatial axes, in order; each must be the
    ``space`` or ``frequency`` dimension of the world axis the affine gives
    it (``xspace`` or ``xfrequency`` and so on).  Default is the names in
    `metadata` for those world axes, if any, and ``xspace``, ``yspace``,
    ``zspace`` otherwise.

    `record_dimension` makes the first dimension the record (unlimited)
    dimension, which MINC1 has and MINC2 does not: True, False, or the
    name of the first dimension.  Default is the record dimension of
    `metadata`, if it is the first dimension of the data.

    `slice_scaling` chooses slice-wise (True) or volume-wise (False)
    image-min / image-max; default is the form of `metadata`, if any, and
    slice-wise otherwise.

    With `vector_dimension` true, the last axis of `data` is the MINC
    ``vector_dimension`` (such as the components of RGB data).  Default is
    true if `metadata` ends with a vector dimension, and `data` has as many
    axes as `metadata`.

    With `metadata`, a :class:`~nibabel.minc_meta.MincMeta`, the info
    variables, and the global, image and dimension attributes are copied,
    except for those the writer works out from the data and the affine: the
    geometry (``start``, ``step``, ``direction_cosines``, ``length``,
    ``spacing``), the scaling (``valid_range`` and the image-min /
    image-max variables) and the attributes that describe the file
    structure (``dimorder``, ``signtype`` and the MINC1 links between
    variables).  The default history line is appended to the history of
    `metadata`, as libminc tools do.

    Returns
    -------
    layout : Layout
    """
    data = np.asanyarray(data)
    if vector_dimension is None:
        vector_dimension = (
            metadata is not None
            and metadata.dimorder[-1:] == (VECTOR_NAME,)
            and len(metadata.dimorder) == data.ndim
        )
    ndim = data.ndim - bool(vector_dimension)
    if ndim not in (3, 4):
        vector = ' and a vector dimension' if vector_dimension else ''
        raise MincWriteError(f'Can only write 3D or 4D data{vector}, not {data.ndim}D')
    out_dtype = check_dtype(data, dtype, dtypes)
    names, starts, steps, cosines = affine_to_dimensions(affine)
    # Slice-wise scaling, as libminc writes by default, or as the source.
    # Scaling does not vary over the vector dimension, as in libminc.
    nscale = ndim - 2
    if slice_scaling is None:
        if metadata is not None and metadata.scaling_dims is not None:
            nscale = min(nscale, len(metadata.scaling_dims))
    elif not slice_scaling:
        nscale = 0

    if labels is None and metadata is not None and metadata.labels:
        labels = dict(metadata.labels)
    if labels:
        labels = {int(value): str(name) for value, name in labels.items()}
        if out_dtype.kind not in 'iu':
            raise MincWriteError(f'Label volumes need an integer type, not {out_dtype}')
        if len(set(labels.values())) != len(labels):
            raise MincWriteError('Label names must be unique')
        limits = np.iinfo(out_dtype)
        stored = np.asarray(data)
        if stored.dtype.kind not in 'iu' and not np.array_equal(stored, np.rint(stored)):
            raise MincWriteError('Label volumes need integer data')
        if stored.size and (stored.min() < limits.min or stored.max() > limits.max):
            raise MincWriteError(f'Label values do not fit in {out_dtype}')
        # Labels are stored as they are
        scaling = (
            np.full(data.shape[:nscale], float(limits.min)),
            np.full(data.shape[:nscale], float(limits.max)),
            (float(limits.min), float(limits.max)),
        )
    if out_dtype.kind == 'f':
        raw = data.astype(out_dtype)
        image_min, image_max, valid_range = _float_ranges(raw, nscale)
    elif out_dtype.kind == 'c':
        # Complex data are not scaled, as floating point data; the range is
        # that of the real and imaginary parts
        raw = data.astype(out_dtype)
        parts = np.stack([raw.real, raw.imag], axis=-1)
        image_min, image_max, valid_range = _float_ranges(parts, nscale)
    else:
        if scaling is None:
            image_min, image_max, valid_range = _choose_scaling(data, out_dtype, nscale)
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
        old_history = '' if metadata is None else metadata.history
        if old_history:
            if not old_history.endswith('\n'):
                old_history += '\n'
            history = old_history + history
    elif not history.endswith('\n'):
        history += '\n'

    dims = []
    if ndim == 4:
        name = 'time'
        if (
            metadata is not None
            and len(metadata.dimorder) == data.ndim
            and metadata.dimorder[0] not in WORLD_AXES
        ):
            name = metadata.dimorder[0]
        dims.append(_dimension(name, raw.shape[0], metadata, start=time_start, step=time_step))
    spatial_shape = raw.shape[ndim - 3 : ndim]
    names = _world_names(names, metadata, dimension_names)
    for i, name in enumerate(names):
        dims.append(
            _dimension(
                name,
                spatial_shape[i],
                metadata,
                start=starts[i],
                step=steps[i],
                cosine=cosines[:, i],
            )
        )
    if vector_dimension:
        dims.append(_vector_dimension(raw.shape[-1], metadata))
    global_attrs = {'ident': make_ident(), 'minc_version': MINC_VERSION, 'history': history}
    image_attrs = {
        'complete': 'true_',
        'valid_range': np.asarray(valid_range, dtype=np.float64),
        **std_attrs('group________'),
    }
    scaling_attrs = {
        'image-max': std_attrs('var_attribute'),
        'image-min': std_attrs('var_attribute'),
    }
    # libminc tools such as nii2mnc make empty study, patient and
    # acquisition groups
    info = [
        MincVariable(
            name, 'info', std_attrs('group________'), data=np.array(MISSING_INT, np.int32)
        )
        for name in ('study', 'patient', 'acquisition')
    ]
    if out_dtype.kind == 'c':
        # libminc tells complex volumes by their class
        global_attrs['class'] = 'complex'
    if labels:
        global_attrs['class'] = 'label__'
    if metadata is not None:
        _copy_attrs(global_attrs, metadata.global_attrs, FILE_ATTRS)
        _copy_attrs(image_attrs, metadata.image_attrs, STRUCTURE_ATTRS)
        # Follow the source in what it leaves to its default
        if 'complete' not in metadata.image_attrs:
            del image_attrs['complete']
        if 'valid_range' not in metadata.image_attrs and _default_range(
            out_dtype, metadata.image_dtype, valid_range
        ):
            del image_attrs['valid_range']
        # The standard attributes too, where the source has those variables
        for att in std_attrs(''):
            if att not in metadata.image_attrs:
                del image_attrs[att]
        for name, attrs in scaling_attrs.items():
            if name in metadata.scaling_attrs:
                source = metadata.scaling_attrs[name]
                for att in std_attrs(''):
                    if att not in source:
                        del attrs[att]
                _copy_attrs(attrs, source, STRUCTURE_ATTRS)
        info = [var for var in metadata.info.values() if not _is_width_of(var, dims)]
    return Layout(
        raw=raw,
        dims=dims,
        global_attrs=global_attrs,
        image_attrs=image_attrs,
        image_min=np.asarray(image_min, dtype=np.float64),
        image_max=np.asarray(image_max, dtype=np.float64),
        scaling_attrs=scaling_attrs,
        info=info,
        metadata=metadata,
        labels=labels,
        record_dimension=_record_dimension(record_dimension, dims, metadata),
    )


def _default_range(out_dtype, src_dtype, valid_range):
    """True if `valid_range` is what a reader assumes without the attribute

    That is the range of integer `out_dtype`, if the source had the same
    type.  For floating point data, readers do not use the valid range.
    """
    if src_dtype is None or out_dtype != np.dtype(src_dtype):
        return False
    if out_dtype.kind in 'fc':
        return True
    info = np.iinfo(out_dtype)
    return tuple(valid_range) == (info.min, info.max)


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
