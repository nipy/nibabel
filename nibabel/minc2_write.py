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

Usage
-----
``write_minc2(data, affine, filename, metadata=img.minc_meta)`` writes `data` with
the voxel to world `affine` and copies the metadata of a MINC1 or MINC2
image read by nibabel (see :mod:`nibabel.minc_meta`): its info variables,
global, image and dimension attributes and history, to which a line is
appended.  The geometry and the scaling always come from `data` and
`affine`.  :class:`~nibabel.minc2.Minc2Image` saves through this function.

Layout
------
The file has a single group ``/minc-2.0`` with the attributes ``ident``,
``minc_version`` and ``history``, and the following members:

* ``dimensions/{x,y,z}space`` (and ``dimensions/time`` for 4D images):
  regularly spaced dimension variables with ``start``, ``step``,
  ``direction_cosines`` (spatial dimensions only), ``spacing='regular__'``,
  ``alignment`` (``'centre'``, or ``'start_'`` for time, as in libminc),
  ``length`` and ``units``;
* ``image/0/image``: the voxel data, with ``dimorder`` and ``valid_range``;
* ``image/0/image-min``, ``image/0/image-max``: the real value range;
* ``info``: the info variables of the ``metadata``, if given; otherwise
  empty.

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

import io
import os

import numpy as np

from ._minc_write import (
    MINC_VERSION,
    SUPPORTED_DTYPES,
    VECTOR_NAME,
    WIDTH_SUFFIX,
    MincWriteError,
    affine_to_dimensions,
    build_layout,
    make_history_line,
    make_ident,
    write_replacing,
)

__all__ = (
    'MINC_VERSION',
    'SUPPORTED_DTYPES',
    'Minc2WriteError',
    'affine_to_dimensions',
    'libminc_chunks',
    'make_history_line',
    'make_ident',
    'write_minc2',
)


#: Error when writing a MINC2 file
Minc2WriteError = MincWriteError


def _string_attr(obj, name, value):
    """Set ASCII string attribute as a null-terminated fixed length string"""
    import h5py

    value = value.encode('ascii')
    tid = h5py.h5t.C_S1.copy()
    tid.set_size(len(value) + 1)
    tid.set_strpad(h5py.h5t.STR_NULLTERM)
    obj.attrs.create(name, np.bytes_(value), dtype=h5py.Datatype(tid))


def _set_attrs(obj, attrs):
    """Set attributes `attrs` of HDF5 object `obj`"""
    for name, value in attrs.items():
        if isinstance(value, str):
            _string_attr(obj, name, value)
        else:
            obj.attrs.create(name, np.asarray(value))


def write_minc2(
    data,
    affine,
    filename,
    dtype=None,
    history=None,
    scaling=None,
    time_start=None,
    time_step=None,
    metadata=None,
    vector_dimension=None,
    slice_scaling=None,
    compression=None,
    chunks=None,
    labels=None,
    dimension_names=None,
    record_dimension=None,
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
        seeking.  A named file is written beside its final name and renamed
        into place, so that an image can be saved over the file its data
        are read from.
    dtype : numpy dtype specifier, optional
        Stored data type: one of uint8, int16, uint16, int32, float32 and
        float64.  Default is the data type of `data`.
    history : str, optional
        Text of the ``history`` attribute.  Default is a libminc style line,
        ``'<ctime>>>> write_minc2 <filename>'``, appended to the history of
        `metadata`, if any.  A string not ending in a newline gets one.
    scaling : tuple, optional
        ``(image_min, image_max, valid_range)`` for integer output, where
        `image_min` and `image_max` are scalars or arrays covering the first
        0, 1 or 2 axes of `data`, and `valid_range` is ``(vmin, vmax)``.
        Default chooses the scaling as described in the module docstring.
        Ignored for floating point output.
    time_start, time_step : float, optional
        ``start`` and ``step`` of the time dimension of a 4D image.  Default
        is the start and step of `metadata`, if it has a time dimension of
        the same length, and 0 and 1 otherwise.
    metadata : MincMeta, optional
        Metadata to copy, such as ``img.minc_meta`` of an image read from a
        MINC file: the info groups (patient, study, acquisition and others),
        and the global, image and dimension attributes.  The geometry and
        the scaling always come from `data` and `affine`, never from
        `metadata`.  A dimension of `metadata` whose length and geometry
        match keeps its exact ``start``, ``step`` and ``direction_cosines``,
        and the coordinates and widths of irregularly spaced samples, which
        are written with ``spacing='irregular'``.  Otherwise the dimension
        is written regularly spaced.  The writer never makes a dimension
        irregular by itself.

    vector_dimension : bool, optional
        If true, the last axis of `data` is the MINC ``vector_dimension``,
        the fastest varying, such as the components of RGB data; it follows
        the spatial axes.  Default is true if `metadata` ends with a vector
        dimension, and `data` has as many axes as `metadata`.

    slice_scaling : bool, optional
        True for slice-wise image-min / image-max (one value per index of
        all but the two fastest varying spatial dimensions), as libminc
        writes by default, False for a single volume-wise range.  Default
        is the form of `metadata`, if any, and slice-wise otherwise.
        Ignored if `scaling` is given.
    compression : None, bool or int, optional
        zlib (gzip) compression level of the image, 0 to 9; True is level 4,
        libminc's default level.  Default is the ``MINC_COMPRESS``
        environment variable, as for libminc, or no compression.
    chunks : None, True or tuple of int, optional
        Chunk shape of the image.  True or None (with compression) gives
        libminc's chunks: whole fastest varying dimensions up to 1,000,000
        bytes per chunk, each at most ``MINC_CHUNKING`` long if that is set
        (and at least 4).  Without compression, None stores the image
        contiguously, as libminc does.
    labels : dict, optional
        Name of each label value, for a label volume (MINC class
        ``label__``), as ``{value: name}``.  libminc stores the names as the
        members of an enumerated HDF5 type of the image.  Default is the
        labels of `metadata`, if any.  The data must be integers that
        `dtype` can hold; they are stored unscaled.

    dimension_names : sequence of str, optional
        Names of the three spatial axes, in order: ``xspace`` or
        ``xfrequency`` for the axis the affine maps to world x, and so on.
        libminc maps spatial frequency dimensions to world axes as it does
        spatial ones.  Default is the names of `metadata` for those world
        axes, if any, and the ``space`` names otherwise.

    record_dimension : bool or str, optional
        MINC2 has no record (unlimited) dimensions, and libminc cannot write
        one to MINC2 (``mincconvert -2`` of such a MINC1 file fails), so a
        record dimension raises an error.  It comes by default from
        `metadata` of a MINC1 file with one; pass False to write that
        dimension as an ordinary one.
    Returns
    -------
    None
    """
    import h5py

    target = filename if isinstance(filename, (str, os.PathLike)) else '<file object>'
    layout = build_layout(
        data,
        affine,
        dtype=dtype,
        history=history,
        scaling=scaling,
        time_start=time_start,
        time_step=time_step,
        metadata=metadata,
        vector_dimension=vector_dimension,
        slice_scaling=slice_scaling,
        dimension_names=dimension_names,
        record_dimension=record_dimension,
        labels=labels,
        command=f'write_minc2 {os.fspath(target)}',
    )

    if layout.record_dimension is not None:
        raise MincWriteError(
            f'MINC2 has no record (unlimited) dimension for {layout.record_dimension}, and '
            'libminc cannot write one to MINC2; pass record_dimension=False to write it as '
            'an ordinary dimension'
        )
    storage = _storage(layout.raw, compression, chunks)

    def write(h5f):
        _write_groups(h5f, layout, storage)

    if isinstance(filename, (str, os.PathLike)):

        def write_file(path):
            with h5py.File(path, 'w') as h5f:
                write(h5f)

        write_replacing(filename, write_file)
    else:
        # Write to memory first, as HDF5 may not create a new file in a file
        # object that it has open for reading
        buffer = io.BytesIO()
        with h5py.File(buffer, 'w') as h5f:
            write(h5f)
        filename.seek(0)
        filename.write(buffer.getvalue())
        filename.truncate()


def _write_range_var(group, name, values, dim_names, attrs):
    values = np.asarray(values, dtype=np.float64)
    var = group.create_dataset(name, data=values)
    if values.ndim:
        attrs = {'dimorder': ','.join(dim_names[: values.ndim]), **attrs}
    _set_attrs(var, attrs)


def _write_info(info, var):
    """Write info variable `var` (a ``MincVariable``) to the info group"""
    if var.is_group:
        obj = info.require_group(var.name)
    else:
        obj = info.create_dataset(var.name, data=var.data)
    attrs = dict(var.attrs)
    attrs.pop('parent', None)
    attrs.pop('children', None)
    if var.dimensions:
        attrs['dimorder'] = ','.join(var.dimensions)
    _set_attrs(obj, attrs)


def _storage(raw, compression, chunks):
    """h5py dataset options for the image: compression and chunks"""
    if compression is None:
        compression = _env_int('MINC_COMPRESS')
    level = 4 if compression is True else int(compression)
    if not 0 <= level <= 9:
        raise MincWriteError(f'Compression level must be 0 to 9, not {level}')
    if chunks is None and level:
        chunks = True
    if chunks is True:
        chunks = libminc_chunks(raw.shape, raw.dtype.itemsize, _env_int('MINC_CHUNKING'))
    storage = {}
    if chunks is not None and chunks is not False and raw.size:
        storage['chunks'] = tuple(int(c) for c in chunks)
    if level:
        storage.update(compression='gzip', compression_opts=level)
    return storage


def _env_int(name):
    try:
        return int(os.environ.get(name, '0'))
    except ValueError:
        return 0


def libminc_chunks(shape, itemsize, chunk_length=0):
    """Chunk shape libminc chooses for compressed data

    As ``hdf_convenience.c`` of libminc: from the fastest varying
    dimension, whole dimensions as long as the chunk stays below
    1,000,000 bytes, then as much of the next dimension as fits, and one
    sample of the others.  A `chunk_length` of 4 or more limits every
    dimension.

    Parameters
    ----------
    shape : tuple of int
        Data shape.
    itemsize : int
        Bytes per value.
    chunk_length : int, optional
        ``MINC_CHUNKING``: largest chunk length per dimension, if 4 or more.

    Returns
    -------
    chunks : tuple of int
    """
    buffer_size = 1000000
    chunks = [0] * len(shape)
    size = 1
    for i in range(len(shape) - 1, -1, -1):
        if buffer_size > shape[i] * size * itemsize:
            chunks[i] = shape[i]
        else:
            chunks[i] = min(shape[i], buffer_size // (size * itemsize))
        chunks[i] = max(chunks[i], 1)
        size *= chunks[i]
    if chunk_length >= 4:
        chunks = [min(c, chunk_length) for c in chunks]
    return tuple(chunks)


def _write_groups(h5f, layout, storage=None):
    """Write the ``/minc-2.0`` group of `layout` to open ``h5f``"""
    import h5py

    minc = h5f.create_group('minc-2.0')
    _set_attrs(minc, layout.global_attrs)
    dims = minc.create_group('dimensions')
    for dim in layout.dims:
        attrs = dict(dim.attrs)
        if dim.data is None:
            # libminc's MINC2 functions store the vector dimension as double
            dtype = 'f8' if dim.name == VECTOR_NAME else 'i4'
            var = dims.create_dataset(dim.name, shape=(), dtype=dtype)
        else:
            var = dims.create_dataset(dim.name, data=dim.data)
            if dim.data.ndim:
                attrs['dimorder'] = dim.name
        _set_attrs(var, attrs)
        if dim.widths is not None:
            group = minc.require_group(dim.width_group)
            var = group.create_dataset(dim.name + WIDTH_SUFFIX, data=dim.widths)
            # libminc reads datasets in /minc-2.0/dimensions as dimensions,
            # with a length
            attrs = {'dimorder': dim.name, **dim.width_attrs, 'length': np.uint32(dim.length)}
            _set_attrs(var, attrs)
    info = minc.require_group('info')
    for var in layout.info:
        _write_info(info, var)
    group = minc.create_group('image').create_group('0')
    raw = layout.raw
    if raw.dtype.kind == 'c':
        # libminc stores complex values as a compound of real and imag
        part = raw.real.dtype
        raw = raw.view(np.dtype([('real', part), ('imag', part)]))
    if layout.labels:
        # libminc stores label names as an enumerated type of the image
        enum = {name: value for value, name in sorted(layout.labels.items())}
        raw = raw.astype(h5py.enum_dtype(enum, basetype=raw.dtype))
    image = group.create_dataset('image', data=raw, **(storage or {}))
    _set_attrs(image, {'dimorder': ','.join(layout.dimorder), **layout.image_attrs})
    for name, values in (('image-max', layout.image_max), ('image-min', layout.image_min)):
        _write_range_var(group, name, values, layout.dimorder, layout.scaling_attrs[name])
