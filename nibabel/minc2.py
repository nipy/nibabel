# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Preliminary MINC2 support

Use with care; I haven't tested this against a wide range of MINC files.

Writing goes through :func:`nibabel.minc2_write.write_minc2`; see that module
for the layout, axis order and scaling of written files.

If you have a file that isn't read correctly, please send an example.

Test reading with something like::

    import nibabel as nib
    img = nib.load('my_funny.mnc')
    data = img.get_fdata()
    print(data.mean())
    print(data.max())
    print(data.min())

and compare against command line output of::

    mincstats my_funny.mnc
"""

import warnings

import numpy as np

from .minc1 import Minc1File, Minc1Image, MincError, MincHeader, MincImageArrayProxy
from .minc2_write import write_minc2
from .minc_meta import MincMeta


class Hdf5Bunch:
    """Make object for accessing attributes of variable"""

    def __init__(self, var):
        for name, value in var.attrs.items():
            setattr(self, name, value)


class Minc2File(Minc1File):
    """Class to wrap MINC2 format file

    Although it has some of the same methods as a ``Header``, we use
    this only when reading a MINC2 file, to pull out useful header
    information, and for the method of reading the data out
    """

    def __init__(self, mincfile):
        self._mincfile = mincfile
        minc_part = mincfile['minc-2.0']
        # The whole image is the first of the entries in 'image'
        image = minc_part['image']['0']
        self._image = image['image']
        self._dim_names = self._get_dimensions(self._image)
        dimensions = minc_part['dimensions']
        self._dims = [Hdf5Bunch(dimensions[s]) for s in self._dim_names]
        # Irregularly spaced dimensions have the coordinate of each sample
        # https://en.wikibooks.org/wiki/MINC/Reference/MINC2.0_File_Format_Reference#Dimension_variable_attributes
        self._coords = {}
        for name, dim in zip(self._dim_names, self._dims):
            # "If this attribute is absent, a value of regular__ should be assumed."
            spacing = getattr(dim, 'spacing', b'regular__')
            if spacing == b'irregular' and dimensions[name].ndim == 1:
                self._coords[name] = np.array(dimensions[name][()], dtype=np.float64)
            elif spacing not in (b'regular__', b'irregular'):
                warnings.warn(f'Invalid spacing declaration: {spacing}; assuming regular')

        self._spatial_dims = [name for name in self._dim_names if name.endswith('space')]
        self._check_irregular()
        self._image_max = image['image-max']
        self._image_min = image['image-min']

    def _get_dimensions(self, var):
        # Dimensions for a particular variable
        # Differs for MINC1 and MINC2 - see:
        # https://en.wikibooks.org/wiki/MINC/Reference/MINC2.0_File_Format_Reference#Associating_HDF5_dataspaces_with_MINC_dimensions
        try:
            dimorder = var.attrs['dimorder'].decode()
        except KeyError:  # No specified dimensions
            return []
        # The dimension name list must contain only as many entries
        # as the variable has dimensions. This reduces errors when an
        # unnecessary dimorder attribute is left behind.
        return dimorder.split(',')[: len(var.shape)]

    def get_data_dtype(self):
        return self._image.dtype

    def get_data_shape(self):
        return self._image.shape

    def _get_valid_range(self):
        """Return valid range for image data

        The valid range can come from the image 'valid_range' or
        failing that, from the data type range
        """
        ddt = self.get_data_dtype()
        info = np.iinfo(ddt.type)
        try:
            valid_range = self._image.attrs['valid_range']
        except (AttributeError, KeyError):
            valid_range = [info.min, info.max]
        else:
            if valid_range[0] < info.min or valid_range[1] > info.max:
                raise ValueError('Valid range outside input data type range')
        return np.asarray(valid_range, dtype=np.float64)

    def _get_scalar(self, var):
        """Get scalar value from HDF5 scalar"""
        return var[()]

    def _get_array(self, var):
        """Get array from HDF5 array"""
        return np.asanyarray(var)

    def get_scaled_data(self, sliceobj=()):
        """Return scaled data for slice definition `sliceobj`

        Parameters
        ----------
        sliceobj : tuple, optional
            slice definition. If not specified, return whole array

        Returns
        -------
        scaled_arr : array
            array from minc file with scaling applied
        """
        if sliceobj == ():
            raw_data = np.asanyarray(self._image)
        else:  # Try slicing into the HDF array (maybe it's possible)
            try:
                raw_data = self._image[sliceobj]
            except (ValueError, TypeError):
                raw_data = np.asanyarray(self._image)[sliceobj]
            else:
                raw_data = np.asanyarray(raw_data)
        return self._normalize(raw_data, sliceobj)


class Minc2Header(MincHeader):
    @classmethod
    def may_contain_header(klass, binaryblock):
        return binaryblock[:4] == b'\211HDF'


class Minc2Image(Minc1Image):
    """Class for MINC2 images

    The MINC2 image class uses the default header type, rather than a
    specific MINC header type - and reads the relevant information from
    the MINC file on load.
    """

    # MINC2 does not do compressed whole files
    _compressed_suffixes = ()
    header_class = Minc2Header
    header: Minc2Header
    rw = True

    @classmethod
    def from_file_map(klass, file_map, *, mmap=True, keep_file_open=None):
        # Import of h5py might take awhile for MPI-enabled builds
        # So we are importing it here "on demand"
        import h5py  # type: ignore[import]

        holder = file_map['image']
        source = holder.fileobj if holder.fileobj is not None else holder.filename
        if source is None:
            raise MincError('MINC2 needs filename or file object for load')
        h5file = h5py.File(source, 'r')
        minc_file = Minc2File(h5file)
        minc_meta = MincMeta.from_hdf5(h5file)
        affine = minc_file.get_affine()
        if affine.shape != (4, 4):
            raise MincError('Image does not have 3 spatial dimensions')
        data_dtype = minc_file.get_data_dtype()
        shape = minc_file.get_data_shape()
        zooms = minc_file.get_zooms()
        header = klass.header_class(data_dtype, shape, zooms)
        data = klass.ImageArrayProxy(minc_file)
        img = klass(data, affine, header, extra=None, file_map=file_map)
        img.minc_meta = minc_meta
        return img

    def to_file_map(self, file_map=None, dtype=None):
        """Write image to `file_map` or contained ``self.file_map``

        Parameters
        ----------
        file_map : None or mapping, optional
           files mapping.  If None (default) use object's ``file_map``
           attribute instead
        dtype : dtype-like, optional
           The on-disk data type to coerce the data array.  Default is the
           header data type.

        Notes
        -----
        The data array axes are the MINC dimensions, slowest varying first;
        a 4D array has time on axis 0, as the reader returns it.  If the
        data are still the unmodified proxy of a MINC file, and `dtype` is
        that file's integer type, the file's image-min / image-max scaling
        is kept, so that the stored values are written back unchanged.  See
        :mod:`nibabel.minc2_write` for the scaling chosen otherwise.

        If the image has a metadata snapshot, ``self.minc_meta``, it is
        passed to :func:`~nibabel.minc2_write.write_minc2`, which copies the
        info variables and the global, image and dimension attributes, and
        appends to the history.  If the data shape or the affine changed
        since the image was read, so that a dimension of the snapshot no
        longer matches, only the geometry of that dimension is dropped
        from the snapshot: its exact start, step and direction cosines,
        and the coordinates and widths of irregular samples.  The dimension
        is then written regularly spaced, with the geometry of the data and
        the affine, and keeps its other attributes.
        """
        if file_map is None:
            file_map = self.file_map
        out_dtype = np.dtype(self.get_data_dtype() if dtype is None else dtype)
        scaling = _proxy_scaling(self.dataobj, out_dtype)
        data = np.asanyarray(self.dataobj)
        holder = file_map['image']
        target = holder.fileobj if holder.fileobj is not None else holder.filename
        write_minc2(
            data,
            self.affine,
            target,
            dtype=out_dtype,
            scaling=scaling,
            metadata=self.minc_meta,
        )
        self.file_map = file_map


def _proxy_scaling(dataobj, out_dtype):
    """Scaling of the MINC file behind `dataobj`, if any

    Returns
    -------
    scaling : None or tuple
        ``(image_min, image_max, valid_range)`` of the source file, if
        `dataobj` is a MINC array proxy and `out_dtype` is its integer
        storage type; None otherwise.
    """
    if not isinstance(dataobj, MincImageArrayProxy):
        return None
    minc_file = dataobj.minc_file
    in_dtype = minc_file.get_data_dtype()
    if out_dtype.kind == 'f' or in_dtype.newbyteorder('=') != out_dtype.newbyteorder('='):
        return None
    image_min = minc_file._image_min
    image_max = minc_file._image_max
    if minc_file._get_dimensions(image_min):
        image_min = np.array(minc_file._get_array(image_min), dtype=np.float64)
        image_max = np.array(minc_file._get_array(image_max), dtype=np.float64)
    else:
        image_min = float(minc_file._get_scalar(image_min))
        image_max = float(minc_file._get_scalar(image_max))
    valid_range = tuple(minc_file._get_valid_range())
    return image_min, image_max, valid_range


load = Minc2Image.from_filename
