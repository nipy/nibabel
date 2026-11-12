# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Read-only snapshot of the metadata of a MINC1 or MINC2 file

The MINC readers keep a :class:`MincMeta` as ``img.minc_meta``, so that the
writers can carry the metadata of a file through a load and save.

MINC1 (NetCDF) and MINC2 (HDF5) files hold the same information in two
layouts.  The snapshot uses the MINC1 model of libminc, which libminc also
uses to read and write MINC2 files through its MINC1 interface: a file has
global attributes and variables, each variable has attributes, and may have
data over named dimensions.  libminc maps this model to MINC2 as follows:

* the global attributes are the attributes of the ``/minc-2.0`` group;
* the ``image``, ``image-min`` and ``image-max`` variables are datasets in
  ``/minc-2.0/image/0``;
* the dimension variables (and the dimension width variables) are datasets
  in ``/minc-2.0/dimensions``;
* all other variables, such as ``patient``, ``study`` and ``acquisition``,
  are datasets in ``/minc-2.0/info``;
* the dimensions of a dataset are in its ``dimorder`` attribute;
* MINC1 attributes that only link variables (``parent``, ``children`` and the
  ``image-min`` / ``image-max`` pointers), the ``rootvariable`` and the
  ``signtype`` of the image (which HDF5 holds in the data type) are not
  stored in MINC2.

The snapshot holds the attributes and data as stored, without the voxel data
and the image-min / image-max values; the writers decide what to copy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

#: Names of the standard MINC dimensions
STANDARD_DIMENSIONS = (
    'xspace',
    'yspace',
    'zspace',
    'time',
    'xfrequency',
    'yfrequency',
    'zfrequency',
    'tfrequency',
    'vector_dimension',
)

#: Suffix of dimension width variables
WIDTH_SUFFIX = '-width'

#: World axis (0, 1, 2 for x, y, z) of the dimensions that libminc maps to
#: world space: the spatial dimensions and the spatial frequency dimensions
WORLD_AXES = {
    'xspace': 0,
    'yspace': 1,
    'zspace': 2,
    'xfrequency': 0,
    'yfrequency': 1,
    'zfrequency': 2,
}

_IMAGE_VARIABLES = ('image', 'image-min', 'image-max')

_EMPTY: Mapping = MappingProxyType({})


def _read_only(arr):
    """Read-only copy of `arr`, in native byte order"""
    arr = np.array(arr)
    if arr.dtype.fields is None and arr.dtype.kind in 'biufc':
        arr = arr.astype(arr.dtype.newbyteorder('='))
    arr.setflags(write=False)
    return arr


def minc1_dtype(var):
    """Data type of MINC1 variable `var`, from its type and ``signtype``

    Without a valid ``signtype``, bytes are unsigned and other integers
    signed, as in libminc.
    """
    dtype = np.dtype(var.data.dtype).newbyteorder('=')
    if dtype.kind != 'i':
        return dtype
    signtype = normalize_attr(var._attributes.get('signtype', ''))
    if signtype == 'unsigned' or (signtype != 'signed__' and dtype.itemsize == 1):
        return np.dtype(f'u{dtype.itemsize}')
    return dtype


def normalize_attr(value):
    """Return attribute `value` as ``str`` or as a read-only array

    Text (``bytes``, ``str``, or arrays of them) becomes ``str``, without
    trailing nulls; a list of strings is joined.  Numbers become a read-only
    array, with zero dimensions if there is a single value, as MINC1
    (NetCDF) attributes have no scalar form.
    """
    if isinstance(value, bytes):
        return value.decode('latin-1').rstrip('\x00')
    if isinstance(value, str):
        return value.rstrip('\x00')
    arr = np.asarray(value)
    if arr.dtype.kind in 'SU' or arr.dtype == object:
        return ''.join(normalize_attr(v) for v in arr.reshape(-1).tolist())
    if arr.size == 1:
        arr = arr.reshape(())
    return _read_only(arr)


def _attr_mapping(attrs):
    return MappingProxyType({str(name): normalize_attr(value) for name, value in attrs.items()})


@dataclass(frozen=True)
class MincVariable:
    """A MINC variable other than the image: its attributes and small data

    Attributes
    ----------
    name : str
        Variable name.  Entries of nested MINC2 info groups have their path
        relative to ``/minc-2.0/info``, such as ``'dicom/0x0008'``.
    group : str
        Where MINC2 keeps the variable: ``'dimensions'`` or ``'info'``; or
        ``'root'`` for the MINC1 ``rootvariable``, which MINC2 does not
        have.
    attrs : mapping
        Attributes, with values as from :func:`normalize_attr`.
    dimensions : tuple of str
        Names of the dimensions of `data`.
    data : None or ndarray
        Read-only copy of the data, or None for MINC2 groups.
    is_group : bool
        True for a MINC2 HDF5 group (with attributes but no data).
    """

    name: str
    group: str
    attrs: Mapping[str, Any] = _EMPTY
    dimensions: tuple[str, ...] = ()
    data: np.ndarray | None = None
    is_group: bool = False


@dataclass(frozen=True)
class MincMeta:
    """Read-only snapshot of the metadata of a MINC1 or MINC2 file

    Attributes
    ----------
    format : str
        ``'MINC1'`` or ``'MINC2'``.
    dimorder : tuple of str
        Dimensions of the image, slowest varying first.
    dim_lengths : mapping
        Length of each dimension of the file.
    global_attrs : mapping
        Global attributes (MINC2: attributes of ``/minc-2.0``), including
        ``history`` and ``ident``.
    image_attrs : mapping
        Attributes of the image variable.
    image_dtype : numpy dtype
        Stored data type of the image.
    scaling_dims : tuple of str
        Dimensions of the image-min and image-max variables, or None if the
        file has none.
    scaling_attrs : mapping
        Attributes of the ``image-max`` and ``image-min`` variables, by name.
    variables : mapping
        :class:`MincVariable` of each variable other than the image,
        image-min and image-max, including the MINC1 ``rootvariable`` (in
        group ``'root'``).
    labels : None or mapping
        For MINC2 label volumes, the name of each label value, as libminc
        stores them in an enumerated HDF5 type.
    record_dimension : None or str
        The record (unlimited) dimension of a MINC1 file, if any.  MINC2
        has none.
    """

    format: str
    dimorder: tuple[str, ...]
    dim_lengths: Mapping[str, int] = _EMPTY
    global_attrs: Mapping[str, Any] = _EMPTY
    image_attrs: Mapping[str, Any] = _EMPTY
    image_dtype: np.dtype | None = None
    scaling_dims: tuple[str, ...] | None = None
    scaling_attrs: Mapping[str, Mapping[str, Any]] = _EMPTY
    variables: Mapping[str, MincVariable] = _EMPTY
    labels: Mapping[int, str] | None = None
    record_dimension: str | None = None

    @property
    def history(self):
        """Text of the global ``history`` attribute, or ``''``"""
        history = self.global_attrs.get('history', '')
        return history if isinstance(history, str) else ''

    @property
    def dimensions(self):
        """Mapping of image dimension name to its :class:`MincVariable`

        Dimensions without a variable, such as ``vector_dimension`` in MINC1,
        are left out.
        """
        return MappingProxyType(
            {name: self.variables[name] for name in self.dimorder if name in self.variables}
        )

    @property
    def info(self):
        """Mapping of name to :class:`MincVariable` of the info variables"""
        return MappingProxyType(
            {name: var for name, var in self.variables.items() if var.group == 'info'}
        )

    @classmethod
    def from_netcdf(klass, ncfile):
        """Snapshot of an open MINC1 ``netcdf_file``"""
        image = ncfile.variables['image']
        scaling_dims = None
        if 'image-max' in ncfile.variables:
            scaling_dims = tuple(ncfile.variables['image-max'].dimensions)
        variables = {}
        for name, var in ncfile.variables.items():
            if name in _IMAGE_VARIABLES:
                continue
            is_dim = name in ncfile.dimensions or (
                name.endswith(WIDTH_SUFFIX) and name[: -len(WIDTH_SUFFIX)] in ncfile.dimensions
            )
            group = 'root' if name == 'rootvariable' else 'dimensions' if is_dim else 'info'
            variables[name] = MincVariable(
                name=name,
                group=group,
                attrs=_attr_mapping(var._attributes),
                dimensions=tuple(var.dimensions),
                data=_read_only(var.data),
            )
        return klass(
            format='MINC1',
            dimorder=tuple(image.dimensions),
            dim_lengths=MappingProxyType(
                {
                    name: int(ncfile._recs if length is None else length)
                    for name, length in ncfile.dimensions.items()
                }
            ),
            record_dimension=next(
                (name for name, length in ncfile.dimensions.items() if length is None), None
            ),
            global_attrs=_attr_mapping(ncfile._attributes),
            image_attrs=_attr_mapping(image._attributes),
            image_dtype=minc1_dtype(image),
            scaling_dims=scaling_dims,
            scaling_attrs=MappingProxyType(
                {
                    name: _attr_mapping(ncfile.variables[name]._attributes)
                    for name in ('image-max', 'image-min')
                    if name in ncfile.variables
                }
            ),
            variables=MappingProxyType(variables),
        )

    @classmethod
    def from_hdf5(klass, h5file):
        """Snapshot of an open MINC2 ``h5py.File``"""
        import h5py

        minc = h5file['minc-2.0']
        group = minc['image/0']
        image = group['image']
        dimorder = _dimorder(image)
        scaling_dims = None
        if 'image-max' in group:
            scaling_dims = _dimorder(group['image-max'])
        variables = {}
        dim_lengths = {}
        for name, dset in minc.get('dimensions', {}).items():
            variables[name] = _hdf5_variable(name, 'dimensions', dset)
            if not name.endswith(WIDTH_SUFFIX) and 'length' in dset.attrs:
                dim_lengths[name] = int(dset.attrs['length'])
        for name, length in zip(dimorder, image.shape):
            dim_lengths[name] = int(length)
        if 'info' in minc:

            def visit(name, obj):
                variables[name] = _hdf5_variable(name, 'info', obj)

            minc['info'].visititems(visit)
        labels = None
        enum = h5py.check_enum_dtype(image.dtype)
        if enum is not None:
            labels = MappingProxyType({int(value): str(key) for key, value in enum.items()})
        return klass(
            format='MINC2',
            dimorder=dimorder,
            dim_lengths=MappingProxyType(dim_lengths),
            global_attrs=_attr_mapping(minc.attrs),
            image_attrs=_attr_mapping(image.attrs),
            image_dtype=np.dtype(image.dtype).newbyteorder('='),
            scaling_dims=scaling_dims,
            scaling_attrs=MappingProxyType(
                {
                    name: _attr_mapping(group[name].attrs)
                    for name in ('image-max', 'image-min')
                    if name in group
                }
            ),
            variables=MappingProxyType(variables),
            labels=labels,
        )


def _dimorder(dset):
    """Dimension names of a MINC2 dataset, from its ``dimorder`` attribute"""
    try:
        dimorder = normalize_attr(dset.attrs['dimorder'])
    except KeyError:
        return ()
    names = tuple(name for name in dimorder.split(',') if name)
    return names[: len(dset.shape)]


def _hdf5_variable(name, group, obj):
    import h5py

    if isinstance(obj, h5py.Group):
        return MincVariable(name=name, group=group, attrs=_attr_mapping(obj.attrs), is_group=True)
    return MincVariable(
        name=name,
        group=group,
        attrs=_attr_mapping(obj.attrs),
        dimensions=_dimorder(obj),
        data=_read_only(obj[()]),
    )
