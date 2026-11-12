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
        command=f'write_minc2 {os.fspath(target)}',
    )

    def write(h5f):
        _write_groups(h5f, layout)

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


def _write_range_var(group, name, values, dim_names):
    values = np.asarray(values, dtype=np.float64)
    var = group.create_dataset(name, data=values)
    attrs = {}
    if values.ndim:
        attrs['dimorder'] = ','.join(dim_names[: values.ndim])
    attrs.update(varid='MINC standard variable', vartype='var_attribute')
    attrs['version'] = 'MINC Version    1.0'
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


def _write_groups(h5f, layout):
    """Write the ``/minc-2.0`` group of `layout` to open ``h5f``"""
    minc = h5f.create_group('minc-2.0')
    _set_attrs(minc, layout.global_attrs)
    dims = minc.create_group('dimensions')
    for dim in layout.dims:
        attrs = dict(dim.attrs)
        if dim.data is None:
            var = dims.create_dataset(dim.name, shape=(), dtype='i4')
        else:
            var = dims.create_dataset(dim.name, data=dim.data)
            if dim.data.ndim:
                attrs['dimorder'] = dim.name
        _set_attrs(var, attrs)
        if dim.widths is not None:
            var = dims.create_dataset(dim.name + WIDTH_SUFFIX, data=dim.widths)
            attrs = {'dimorder': dim.name, 'length': np.uint32(dim.length), **dim.width_attrs}
            _set_attrs(var, attrs)
    info = minc.create_group('info')
    for var in layout.info:
        _write_info(info, var)
    group = minc.create_group('image').create_group('0')
    image = group.create_dataset('image', data=layout.raw)
    attrs = dict(layout.image_attrs)
    attrs = {'complete': attrs.pop('complete'), 'dimorder': ','.join(layout.dimorder), **attrs}
    _set_attrs(image, attrs)
    _write_range_var(group, 'image-max', layout.image_max, layout.dimorder)
    _write_range_var(group, 'image-min', layout.image_min, layout.dimorder)
