# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Write MINC1 (NetCDF) format images

This module writes MINC1 files with nibabel's own NetCDF module,
:mod:`nibabel.externals.netcdf`, so it needs only numpy.  It shares the
choice of dimensions, scaling and metadata with the MINC2 writer
(:mod:`nibabel.minc2_write`); see there for the axis order and the scaling.

See the `MINC1 programmer's guide
<https://en.wikibooks.org/wiki/MINC/Reference/MINC1-programmers-guide>`_.

Layout
------
The layout follows libminc's MINC1 functions (``micreate_std_variable`` and
friends), as written for example by ``nii2mnc``:

* global attributes ``ident``, ``minc_version`` and ``history``;
* a NetCDF dimension for each image dimension, and a dimension variable of
  the same name, of type int without data for regularly spaced dimensions,
  or of type double with the coordinate of each sample for irregularly
  spaced ones.  It has the ``varid``, ``vartype`` and ``version``
  attributes of a MINC standard variable, ``comments`` for spatial
  dimensions, ``spacing``, ``alignment``, ``step``, ``units``, ``start`` and
  ``direction_cosines`` (spatial dimensions only).  Irregularly spaced
  dimensions may have a ``<name>-width`` variable with the width of each
  sample;
* the ``image-max`` and ``image-min`` variables (double), slice-wise by
  default, with ``_FillValue`` and ``parent`` attributes;
* the ``rootvariable``, the parent of the ``patient``, ``study`` and
  ``acquisition`` variables (only with metadata that has them) and of the
  image, whose ``children`` attribute lists them;
* the ``image`` variable with ``parent``, pointers to ``image-max`` and
  ``image-min``, ``signtype`` and ``valid_range``.

NetCDF has no unsigned types; unsigned data are stored in the signed type
of the same size with ``signtype`` set to ``unsigned``.  MINC1 has no 64-bit
integer and no complex types.
"""

import io
import os
import warnings

import numpy as np

from ._minc_write import (
    MISSING_INT,
    WIDTH_SUFFIX,
    MincWriteError,
    build_layout,
    std_attrs,
    write_replacing,
)
from .externals.netcdf import NC_VARIABLE, netcdf_file

#: Data types that can be written to MINC1
MINC1_DTYPES = tuple(
    np.dtype(t)
    for t in (np.uint8, np.int8, np.int16, np.uint16, np.int32, np.uint32, np.float32, np.float64)
)

# MINC1 attributes that link variables, written by this module
_LINK_ATTRS = ('parent', 'children', 'image-min', 'image-max', '_FillValue')
# Largest file that NetCDF classic offsets can address
_CLASSIC_MAX = 2**31 - 1


def write_minc1(
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
    """Write `data` and `affine` to a MINC1 file

    Parameters
    ----------
    data : array-like
        3D array (three spatial axes), or 4D array with time on axis 0 and
        the spatial axes after it, as for
        :func:`~nibabel.minc2_write.write_minc2`.
    affine : array-like, shape (4, 4)
        Voxel to world affine of the spatial axes.
    filename : str, os.PathLike or file-like
        Output file name, or a binary file object open for writing and
        seeking.  A named file is written beside its final name and renamed
        into place.
    dtype : numpy dtype specifier, optional
        Stored data type: one of uint8, int8, int16, uint16, int32, uint32,
        float32 and float64.  Default is the data type of `data`.
    history : str, optional
        Text of the ``history`` attribute.  Default is a libminc style line,
        ``'<ctime>>>> write_minc1 <filename>'``, appended to the history of
        `metadata`, if any.
    scaling : tuple, optional
        ``(image_min, image_max, valid_range)`` for integer output, as for
        :func:`~nibabel.minc2_write.write_minc2`.
    time_start, time_step : float, optional
        ``start`` and ``step`` of the time dimension of a 4D image.
    metadata : MincMeta, optional
        Metadata to copy, from a MINC1 or a MINC2 file, as for
        :func:`~nibabel.minc2_write.write_minc2`.  Nested MINC2 info groups
        and MINC2 labels, which MINC1 cannot hold, are left out with a
        warning.

    Returns
    -------
    None
    """
    data = np.asanyarray(data)
    out_dtype = np.dtype(data.dtype if dtype is None else dtype).newbyteorder('=')
    if out_dtype not in MINC1_DTYPES:
        allowed = ', '.join(str(t) for t in MINC1_DTYPES)
        raise MincWriteError(f'Cannot write data type {out_dtype} to MINC1; use one of {allowed}')
    target = filename if isinstance(filename, (str, os.PathLike)) else '<file object>'
    layout = build_layout(
        data,
        affine,
        dtype=out_dtype,
        dtypes=MINC1_DTYPES,
        history=history,
        scaling=scaling,
        time_start=time_start,
        time_step=time_step,
        metadata=metadata,
        command=f'write_minc1 {os.fspath(target)}',
    )
    contents = _netcdf_bytes(layout)
    if isinstance(filename, (str, os.PathLike)):

        def write_file(path):
            with open(path, 'wb') as fobj:
                fobj.write(contents)

        write_replacing(filename, write_file)
    else:
        filename.seek(0)
        filename.write(contents)
        filename.truncate()


def _nc_value(value):
    """Return attribute `value` in a type NetCDF classic can hold"""
    if isinstance(value, str):
        return value.encode('latin-1')
    arr = np.atleast_1d(np.asarray(value))
    if arr.dtype.kind == 'f':
        return arr.astype('>f4' if arr.dtype.itemsize <= 4 else '>f8')
    if arr.dtype.kind in 'iub':
        info = np.iinfo(np.int32)
        if arr.dtype.itemsize <= 2 and arr.dtype.kind == 'i':
            return arr.astype(f'>i{max(arr.dtype.itemsize, 2)}')
        if arr.size and (arr.min() < info.min or arr.max() > info.max):
            return arr.astype('>f8')
        return arr.astype('>i4')
    raise MincWriteError(f'Cannot write attribute of type {arr.dtype} to MINC1')


def _nc_type(dtype):
    """NetCDF storage type and signtype for numpy `dtype`"""
    dtype = np.dtype(dtype)
    if dtype.kind == 'f':
        return np.dtype(f'f{dtype.itemsize}'), None
    signtype = 'unsigned' if dtype.kind in 'ub' else 'signed__'
    return np.dtype(f'i{dtype.itemsize}'), signtype


def _set_attrs(var, attrs):
    for name, value in attrs.items():
        setattr(var, name, _nc_value(value))


def _source_attrs(layout, name):
    """Attributes of variable `name` of the layout's MINC1 source, or {}"""
    meta = layout.metadata
    if meta is None or meta.format != 'MINC1' or name not in meta.variables:
        return {}
    return meta.variables[name].attrs


class _OrderedNetcdf(netcdf_file):
    """NetCDF file that keeps its variables in the order they were made

    netcdf_file sorts the variables by shape; libminc keeps them in order,
    with the image last.
    """

    def _write_var_array(self):
        if not self.variables:
            return super()._write_var_array()
        self.fp.write(NC_VARIABLE)
        self._pack_int(len(self.variables))
        for name in self.variables:
            self._write_var_metadata(name)
        self.__dict__['_recsize'] = 0
        for name in self.variables:
            self._write_var_data(name)


def _netcdf_bytes(layout):
    """Return the contents of a MINC1 file for `layout`"""
    size = layout.raw.nbytes + sum(
        8 * np.size(dim.data) for dim in layout.dims if dim.data is not None
    )
    buffer = io.BytesIO()
    ncfile = _OrderedNetcdf(buffer, 'w', version=1 if size < _CLASSIC_MAX else 2)
    _set_attrs(ncfile, layout.global_attrs)
    dimorder = layout.dimorder
    for dim in layout.dims:
        ncfile.createDimension(dim.name, dim.length)

    children = []
    for dim in layout.dims:
        _write_dimension(ncfile, layout, dim)
    nscale = layout.image_max.ndim
    for name, values in (('image-max', layout.image_max), ('image-min', layout.image_min)):
        var = ncfile.createVariable(name, 'd', dimorder[:nscale])
        var[...] = values
        attrs = dict(std_attrs('var_attribute'))
        attrs.update(layout.scaling_attrs.get(name, {}))
        attrs['_FillValue'] = np.float64(1.0 if name == 'image-max' else 0.0)
        attrs['parent'] = 'image'
        _set_attrs(var, attrs)
    root = None
    for info in layout.info:
        if '/' in info.name:
            warnings.warn(f'MINC1 cannot hold the nested MINC2 info entry {info.name}; left out')
            continue
        _write_info(ncfile, info)
        children.append(info.name)
        if root is None:
            # libminc makes the root variable with its first child
            root = ncfile.createVariable('rootvariable', 'i', ())
            root[...] = MISSING_INT
    if root is None:
        root = ncfile.createVariable('rootvariable', 'i', ())
        root[...] = MISSING_INT
    children.append('image')
    _set_attrs(root, {**std_attrs('group________'), 'parent': '', 'children': '\n'.join(children)})

    if layout.metadata is not None and layout.metadata.labels:
        warnings.warn('MINC1 cannot hold the label names of a MINC2 label volume; left out')
    nc_dtype, signtype = _nc_type(layout.raw.dtype)
    image = ncfile.createVariable('image', nc_dtype, dimorder)
    image[...] = layout.raw.view(nc_dtype)
    attrs = {'parent': 'rootvariable', **std_attrs('group________')}
    attrs['image-max'] = '--->image-max'
    attrs['image-min'] = '--->image-min'
    extra = dict(layout.image_attrs)
    for name in ('complete', 'valid_range'):
        extra.pop(name, None)
    attrs.update(extra)
    meta = layout.metadata
    if meta is not None and meta.format == 'MINC1' and 'dimorder' in meta.image_attrs:
        attrs['dimorder'] = ','.join(dimorder)
    if 'complete' in layout.image_attrs:
        attrs['complete'] = layout.image_attrs['complete']
    if signtype is not None:
        attrs['signtype'] = signtype
    if 'valid_range' in layout.image_attrs:
        # libminc writes the valid range as double, except for float images
        valid_range = np.asarray(layout.image_attrs['valid_range'], dtype=np.float64)
        if nc_dtype == np.float32:
            valid_range = valid_range.astype(np.float32)
        attrs['valid_range'] = valid_range
    _set_attrs(image, attrs)
    ncfile.flush()
    contents = buffer.getvalue()
    ncfile.close()
    return contents


def _write_dimension(ncfile, layout, dim):
    """Write the dimension variable of `dim`, and its width variable"""
    source = _source_attrs(layout, dim.name)
    if dim.data is not None and dim.data.ndim:
        var = ncfile.createVariable(dim.name, 'd', (dim.name,))
        var[...] = dim.data
    else:
        var = ncfile.createVariable(dim.name, 'i', ())
        var[...] = MISSING_INT
    attrs = {}
    for name, value in dim.attrs.items():
        if name == 'length':
            # MINC1 dimensions have a length of their own; keep the
            # attribute only where the source has it
            if 'length' in source:
                attrs[name] = np.int32(dim.length)
        else:
            attrs[name] = value
    if 'dimorder' in source:
        attrs = {'dimorder': ','.join(var.dimensions), **attrs}
    _set_attrs(var, attrs)
    if dim.widths is not None:
        width = ncfile.createVariable(dim.name + WIDTH_SUFFIX, 'd', (dim.name,))
        width[...] = dim.widths
        _set_attrs(width, dim.width_attrs)


def _write_info(ncfile, info):
    """Write MINC info variable `info` (a ``MincVariable``)"""
    attrs = {'parent': 'rootvariable'}
    attrs.update((k, v) for k, v in info.attrs.items() if k not in _LINK_ATTRS + ('dimorder',))
    data = info.data
    if info.is_group or data is None:
        var = ncfile.createVariable(info.name, 'i', ())
        var[...] = MISSING_INT
    else:
        dims = info.dimensions
        if data.ndim and len(dims) != data.ndim:
            warnings.warn(f'Info variable {info.name} has data without dimensions; left out')
            data, dims = np.array(MISSING_INT, dtype=np.int32), ()
        for name, length in zip(dims, data.shape):
            if name not in ncfile.dimensions:
                ncfile.createDimension(name, length)
        if data.dtype.kind == 'S':
            var = ncfile.createVariable(info.name, 'c', dims)
        else:
            data = _nc_value(data).reshape(data.shape)
            var = ncfile.createVariable(info.name, data.dtype.newbyteorder('='), dims)
        var[...] = data
        if 'dimorder' in info.attrs:
            attrs['dimorder'] = ','.join(dims)
    _set_attrs(var, attrs)
