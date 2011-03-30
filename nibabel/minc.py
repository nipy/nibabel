# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
import numpy as np

from .externals.netcdf import netcdf_file

from .py3k import asbytes, asstr
from .spatialimages import SpatialImage

_dt_dict = {
    ('b','unsigned'): np.uint8,
    ('b','signed__'): np.int8,
    ('c','unsigned'): 'S1',
    ('h','unsigned'): np.uint16,
    ('h','signed__'): np.int16,
    ('i','unsigned'): np.uint32,
    ('i','signed__'): np.int32,
    }

# See http://www.bic.mni.mcgill.ca/software/minc/minc1_format/node15.html
_default_dir_cos = {
    'xspace': [1,0,0],
    'yspace': [0,1,0],
    'zspace': [0,0,1]}


class MincError(Exception):
    pass


class MincFile(object):
    ''' Class to wrap MINC file

    Although it has some of the same methods as a ``Header``, we use
    this only when reading a MINC file, to pull out useful header
    information, and for the method of reading the data out
    '''
    def __init__(self, mincfile):
        self._mincfile = mincfile
        self._image = mincfile.variables['image']
        self._dim_names = self._image.dimensions
        # The code below will error with vector_dimensions.  See:
        # http://www.bic.mni.mcgill.ca/software/minc/minc1_format/node3.html
        # http://www.bic.mni.mcgill.ca/software/minc/prog_guide/node11.html
        self._dims = [self._mincfile.variables[s]
                      for s in self._dim_names]
        # We don't currently support irregular spacing
        # http://www.bic.mni.mcgill.ca/software/minc/minc1_format/node15.html
        for dim in self._dims:
            if dim.spacing != asbytes('regular__'):
                raise ValueError('Irregular spacing not supported')
        self._spatial_dims = [name for name in self._dim_names
                             if name.endswith('space')]

    def get_data_dtype(self):
        typecode = self._image.typecode()
        if typecode == 'f':
            dtt = np.dtype(np.float32)
        elif typecode == 'd':
            dtt = np.dtype(np.float64)
        else:
            signtype = asstr(self._image.signtype)
            dtt = _dt_dict[(typecode, signtype)]
        return np.dtype(dtt).newbyteorder('>')

    def get_data_shape(self):
        return self._image.data.shape

    def get_zooms(self):
        return tuple(
            [float(dim.step) for dim in self._dims])

    def get_affine(self):
        nspatial = len(self._spatial_dims)
        rot_mat = np.eye(nspatial)
        steps = np.zeros((nspatial,))
        starts = np.zeros((nspatial,))
        dim_names = list(self._dim_names) # for indexing in loop
        for i, name in enumerate(self._spatial_dims):
            dim = self._dims[dim_names.index(name)]
            try:
                dir_cos = dim.direction_cosines
            except AttributeError:
                dir_cos = _default_dir_cos[name]
            rot_mat[:, i] = dir_cos
            steps[i] = dim.step
            starts[i] = dim.start
        origin = np.dot(rot_mat, starts)
        aff = np.eye(nspatial+1)
        aff[:nspatial, :nspatial] = rot_mat * steps
        aff[:nspatial, nspatial] = origin
        return aff

    def _get_valid_range(self):
        ''' Return valid range for image data

        The valid range can come from the image 'valid_range' or
        image 'valid_min' and 'valid_max', or, failing that, from the
        data type range
        '''
        ddt = self.get_data_dtype()
        info = np.iinfo(ddt.type)
        try:
            valid_range = self._image.valid_range
        except AttributeError:
            try:
                valid_range = [self._image.valid_min,
                               self._image.valid_max]
            except AttributeError:
                valid_range = [info.min, info.max]
        if valid_range[0] < info.min or valid_range[1] > info.max:
            raise ValueError('Valid range outside input '
                             'data type range')
        return np.asarray(valid_range, dtype=np.float)

    def _normalize(self, data):
        """ Scale image data with recorded scalefactors

        http://www.bic.mni.mcgill.ca/software/minc/prog_guide/node13.html

        MINC normalization uses "image-min" and "image-max" variables to
        map the data from the valid range of the image to the range
        specified by "image-min" and "image-max".

        The "image-max" and "image-min" are variables that describe the
        "max" and "min" of image over some dimensions of "image".

        The usual case is that "image" has dimensions ["zspace",
        "yspace", "xspace"] and "image-max" has dimensions
        ["zspace"].
        """
        ddt = self.get_data_dtype()
        if ddt.type in np.sctypes['float']:
            return data
        # the MINC standard appears to allow the following variables to
        # be undefined.
        # http://www.bic.mni.mcgill.ca/software/minc/minc1_format/node16.html
        # It wasn't immediately obvious what the defaults were.
        image_max = self._mincfile.variables['image-max']
        image_min = self._mincfile.variables['image-min']
        if image_max.dimensions != image_min.dimensions:
            raise MincError('"image-max" and "image-min" do not '
                             'have the same dimensions')
        nscales = len(image_max.dimensions)
        if image_max.dimensions != self._dim_names[:nscales]:
            raise MincError('image-max and image dimensions '
                            'do not match')
        dmin, dmax = self._get_valid_range()
        if nscales == 0:
            imax = np.asarray(image_max)
            imin = np.asarray(image_min)
            sc = (imax-imin) / (dmax-dmin)
            return np.clip(data, dmin, dmax) * sc + (imin - dmin * sc)
        out_data = np.empty(data.shape, np.float)

        def _norm_slice(sdef):
            imax = image_max[sdef]
            imin = image_min[sdef]
            in_data = np.clip(data[sdef], dmin, dmax)
            sc = (imax-imin) / (dmax-dmin)
            return in_data * sc + (imin - dmin * sc)

        if nscales == 1:
            for i in range(data.shape[0]):
                out_data[i] = _norm_slice(i)
        elif nscales == 2:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    out_data[i, j] = _norm_slice((i,j))
        else:
            raise MincError('More than two scaling dimensions')
        return out_data

    def get_scaled_data(self):
        dtype = self.get_data_dtype()
        data =  np.asarray(self._image.data).view(dtype)
        return self._normalize(data)


class MincImage(SpatialImage):
    ''' Class for MINC images

    The MINC image class uses the default header type, rather than a
    specific MINC header type - and reads the relevant information from
    the MINC file on load.
    '''
    files_types = (('image', '.mnc'),)
    _compressed_exts = ('.gz', '.bz2')

    class ImageArrayProxy(object):
        ''' Minc implemention of array proxy protocol

        The array proxy allows us to freeze the passed fileobj and
        header such that it returns the expected data array.
        '''
        def __init__(self, minc_file):
            self.minc_file = minc_file
            self._data = None
            self.shape = minc_file.get_data_shape()

        def __array__(self):
            ''' Cached read of data from file '''
            if self._data is None:
                self._data = self.minc_file.get_scaled_data()
            return self._data

    @classmethod
    def from_file_map(klass, file_map):
        fobj = file_map['image'].get_prepare_fileobj()
        minc_file = MincFile(netcdf_file(fobj))
        affine = minc_file.get_affine()
        if affine.shape != (4, 4):
            raise MincError('Image does not have 3 spatial dimensions')
        data_dtype = minc_file.get_data_dtype()
        shape = minc_file.get_data_shape()
        zooms = minc_file.get_zooms()
        header = klass.header_class(data_dtype, shape, zooms)
        data = klass.ImageArrayProxy(minc_file)
        return  MincImage(data, affine, header, extra=None, file_map=file_map)


load = MincImage.load
