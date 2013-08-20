# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Preliminary MINC2 support

Use with care; I haven't tested this against a wide range of MINC files.

If you have a file that isn't read correctly, please send an example.

Test reading with something like::

    import nibabel as nib
    img = nib.load('my_funny.mnc')
    data = img.get_data()
    print(data.mean())
    print(data.max())
    print(data.min())

and compare against command line output of::

    mincstats my_funny.mnc
"""
import numpy as np

from .optpkg import optional_package
h5py, have_h5py, setup_module = optional_package('h5py')

from .minc1 import Minc1File, Minc1Image, MincError


class Hdf5Bunch(object):
    """ Make object for accessing attributes of variable
    """
    def __init__(self, var):
        for name, value in var.attrs.items():
            setattr(self, name, value)


class Minc2File(Minc1File):
    ''' Class to wrap MINC2 format file

    Although it has some of the same methods as a ``Header``, we use
    this only when reading a MINC2 file, to pull out useful header
    information, and for the method of reading the data out
    '''
    def __init__(self, mincfile):
        self._mincfile = mincfile
        minc_part = mincfile['minc-2.0']
        # The whole image is the first of the entries in 'image'
        image = minc_part['image']['0']
        self._image = image['image']
        self._dim_names = self._get_dimensions(self._image)
        dimensions = minc_part['dimensions']
        self._dims = [Hdf5Bunch(dimensions[s]) for s in self._dim_names]
        # We don't currently support irregular spacing
        # http://en.wikibooks.org/wiki/MINC/Reference/MINC2.0_File_Format_Reference#Dimension_variable_attributes
        for dim in self._dims:
            if dim.spacing != b'regular__':
                raise ValueError('Irregular spacing not supported')
        self._spatial_dims = [name for name in self._dim_names
                              if name.endswith('space')]
        self._image_max = image['image-max']
        self._image_min = image['image-min']

    def _get_dimensions(self, var):
        # Dimensions for a particular variable
        # Differs for MINC1 and MINC2 - see:
        # http://en.wikibooks.org/wiki/MINC/Reference/MINC2.0_File_Format_Reference#Associating_HDF5_dataspaces_with_MINC_dimensions
        return var.attrs['dimorder'].split(',')

    def get_data_dtype(self):
        return self._image.dtype

    def get_data_shape(self):
        return self._image.shape

    def _get_valid_range(self):
        ''' Return valid range for image data

        The valid range can come from the image 'valid_range' or
        failing that, from the data type range
        '''
        ddt = self.get_data_dtype()
        info = np.iinfo(ddt.type)
        try:
            valid_range = self._image.attrs['valid_range']
        except AttributeError:
            valid_range = [info.min, info.max]
        else:
            if valid_range[0] < info.min or valid_range[1] > info.max:
                raise ValueError('Valid range outside input '
                                 'data type range')
        return np.asarray(valid_range, dtype=np.float)

    def get_scaled_data(self):
        data =  np.asarray(self._image)
        return self._normalize(data)


class Minc2Image(Minc1Image):
    ''' Class for MINC2 images

    The MINC2 image class uses the default header type, rather than a
    specific MINC header type - and reads the relevant information from
    the MINC file on load.
    '''
    # MINC2 does not do compressed whole files
    _compressed_exts = ()

    @classmethod
    def from_file_map(klass, file_map):
        holder = file_map['image']
        if holder.filename is None:
            raise MincError('MINC2 needs filename for load')
        minc_file = Minc2File(h5py.File(holder.filename, 'r'))
        affine = minc_file.get_affine()
        if affine.shape != (4, 4):
            raise MincError('Image does not have 3 spatial dimensions')
        data_dtype = minc_file.get_data_dtype()
        shape = minc_file.get_data_shape()
        zooms = minc_file.get_zooms()
        header = klass.header_class(data_dtype, shape, zooms)
        data = klass.ImageArrayProxy(minc_file)
        return klass(data, affine, header, extra=None, file_map=file_map)


load = Minc2Image.load
