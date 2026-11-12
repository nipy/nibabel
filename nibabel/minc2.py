# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Read and write MINC2 format images

:class:`Minc2Image` reads MINC2 (HDF5) files with h5py, and writes them with
:func:`nibabel.minc2_write.write_minc2`; see that module for the layout, axis
order and scaling of written files.  It shares the interface of
:class:`~nibabel.minc1.Minc1Image`::

    import nibabel as nib
    img = nib.load('in.mnc')               # Minc2Image for a MINC2 file
    meta = img.minc_meta                   # read-only metadata snapshot
    nib.save(img, 'out.mnc')               # MINC2, metadata kept
    nib.save(img, 'gz.mnc', compression=True)
    nib.save(nib.Minc1Image.from_image(img), 'v1.mnc')  # convert to MINC1

``img.minc_meta`` is a :class:`~nibabel.minc_meta.MincMeta`; for label
volumes, ``img.minc_meta.labels`` maps label values to names.  Saving passes
it to the writer, and ``from_image`` carries it between
:class:`~nibabel.minc1.Minc1Image` and :class:`Minc2Image`.  ``nib.save``
passes keyword arguments to the writer: ``slice_scaling``, ``compression``,
``chunks`` and ``labels``.

If you have a file that isn't read correctly, please send an example, and
compare with the output of::

    mincstats my_funny.mnc
"""

import warnings

import numpy as np

from .minc1 import Minc1File, Minc1Image, MincError, MincHeader
from .minc2_write import write_minc2
from .minc_meta import MincMeta


class Hdf5Bunch:
    """Make object for accessing attributes of variable"""

    def __init__(self, var):
        if var is None:
            return
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
        # The vector dimension (such as RGB components) may have no dataset
        self._dims = [
            Hdf5Bunch(dimensions[s]) if s in dimensions else Hdf5Bunch(None)
            for s in self._dim_names
        ]
        # Irregularly spaced dimensions have the coordinate of each sample
        # https://en.wikibooks.org/wiki/MINC/Reference/MINC2.0_File_Format_Reference#Dimension_variable_attributes
        self._coords = {}
        for name, dim in zip(self._dim_names, self._dims):
            # "If this attribute is absent, a value of regular__ should be assumed."
            spacing = getattr(dim, 'spacing', b'regular__')
            if spacing == b'irregular' and name in dimensions and dimensions[name].ndim == 1:
                self._coords[name] = np.array(dimensions[name][()], dtype=np.float64)
            elif spacing not in (b'regular__', b'irregular'):
                warnings.warn(f'Invalid spacing declaration: {spacing}; assuming regular')

        self._spatial_dims = [name for name in self._dim_names if name.endswith('space')]
        self._check_irregular()
        self._image_max = image.get('image-max')
        self._image_min = image.get('image-min')

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
        dtype = self._image.dtype
        if _is_complex(dtype):
            return np.dtype(f'c{2 * dtype["real"].itemsize}')
        # The labels of label volumes are in minc_meta, not in the type
        return np.dtype(dtype.str)

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
        if _is_complex(raw_data.dtype):
            raw_data = raw_data['real'] + 1j * raw_data['imag']
        return self._normalize(raw_data, sliceobj)


def _is_complex(dtype):
    """True for libminc's compound type of complex values"""
    return dtype.names == ('real', 'imag')


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

    @staticmethod
    def _write(*args, **kwargs):
        return write_minc2(*args, **kwargs)


load = Minc2Image.from_filename
