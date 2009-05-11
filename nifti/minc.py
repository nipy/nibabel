import numpy as np

from scipy.io.netcdf import netcdf_file as netcdf

from nifti.spatialimages import SpatialImage
from nifti.volumeutils import allopen

_dt_dict = {
    ('b','unsigned'): np.uint8,
    ('b','signed__'): np.int8,
    ('c','unsigned'): 'S1',
    ('h','unsigned'): np.uint16,
    ('h','signed__'): np.int16,
    ('i','unsigned'): np.uint32,
    ('i','signed__'): np.int32,
    }


class netcdf_fileobj(netcdf):
    def __init__(self, fileobj):
        self._buffer = fileobj
        self._parse()


class MINCHeader(object):
    def __init__(self, mincfile, endianness=None, check=True):
        self.endianness = '>'
        self._mincfile = mincfile
        self._image = mincfile.variables['image']
        self._dims = [self._mincfile.variables[s]
                      for s in self._image.dimensions]
        if check:
            self.check_fix()

    @classmethod
    def from_fileobj(klass, fileobj, endianness=None, check=True):
        ncdf_obj = netcdf_fileobj(fileobj)
        return klass(ncdf_obj, endianness, check)

    def check_fix(self):
        for dim in self._dims:
            if dim.spacing != 'regular__':
                raise ValueError('Irregular spacing not supported')
        image_max = self._mincfile.variables['image-max']
        image_min = self._mincfile.variables['image-min']
        if image_max.dimensions != image_min.dimensions:
            raise ValueError('"image-max" and "image-min" do not '
                             'have the same dimensions')

    def get_data_shape(self):
        return self._image.shape
        
    def get_data_dtype(self):
        typecode = self._image.typecode()
        if typecode == 'f':
            dtt = np.dtype(np.float32)
        elif typecode == 'd':
            dtt = np.dtype(np.float64)
        else:
            signtype = self._image.signtype
            dtt = _dt_dict[(typecode, signtype)]
        return np.dtype(dtt).newbyteorder('>')

    def get_zooms(self):
        return tuple(
            [float(dim.step) for dim in self._dims])

    def get_best_affine(self):
        zooms = self.get_zooms()
        rot_mat = np.eye(3)
        starts = np.zeros((3,))
        for i, dim in enumerate(self._dims):
            rot_mat[:,i] = dim.direction_cosines
            starts[i] = dim.start
        origin = np.dot(rot_mat, starts)
        rz = rot_mat * zooms
        aff = np.eye(4)
        aff[:3,:3] = rot_mat * zooms
        aff[:3,3] = origin
        return aff

    def get_unscaled_data(self):
        return np.asarray(self._image)

    def _normalize(self, data):
        """
        MINC normalization:

        Otherwise, it uses "image-min" and "image-max" variables
        to map the data from the valid range of the NC_TYPE to the
        range specified by "image-min" and "image-max".

        If self.norm_range is not None, it is used in place of the
        builtin default valid ranges of the NC_TYPEs. If the NC_TYPE
        is NC_FLOAT or NC_DOUBLE, then the transformation is only done if 
        self.norm_range is not None, otherwise the data is untransformed.

        The "image-max" and "image-min" are variables that describe the
        "max" and "min" of image over some dimensions of "image".

        The usual case is that "image" has dimensions ["zspace", "yspace", "xspace"]
        and "image-max" has dimensions ["zspace"]. In this case, the
        normalization is defined by the following transformation:

        for i in range(d.shape[0]):
            d[i] = (clip((d - norm_range[i]).astype(float) / 
                         (norm_range[i] - norm_range[i]), 0, 1) * 
                         (image_max[i] - image_min[i]) + image_min[i])

        """
        ddt = self.get_data_dtype()
        if ddt.type in np.sctypes['float']:
            return data
        info = np.iinfo(ddt.type)
        vrange = [info.min, info.max]
        image_max = self._mincfile.variables['image-max']
        image_min = self._mincfile.variables['image-min']
        imdims = self._image.dimensions
        dims = self._mincfile.dimensions
        axes = [list(imdims).index(d) for d in image_max.dimensions]  
        shape = [dims[d] for d in image_max.dimensions]
        indices = np.indices(shape)
        indices.shape = (indices.shape[0], np.product(indices.shape[1:]))

        out_data = np.zeros(data.shape, np.float)
        vdiff = float(vrange[1] - vrange[0])
        for index in indices.T:
            slice_ = []
            aslice_ = []
            iaxis = 0
	    for idim, dim in enumerate(imdims):
                if idim not in axes:
                    slice_.append(slice(0,dims[dim],1))
                else:
                    slice_.append(slice(index[iaxis], index[iaxis]+1,1))
                    aslice_.append(slice(index[iaxis], index[iaxis]+1,1))
                    iaxis += 1
            try:
                _image_min = image_min[aslice_]
                _image_max = image_max[aslice_]
            except IndexError:
                _image_min = image_min.getValue()
                _image_max = image_max.getValue()
            out_data[slice_] = np.clip(
                (data[slice_] - vrange[0]).astype(float) / vdiff,
                0.0, 1.1) * (_image_max-_image_min) + image_min
	return out_data

    def get_scaled_data(self):
        return self._normalize(self.get_unscaled_data())
    

class MINCImage(SpatialImage):
    _header_maker = MINCHeader
    
    def _set_header(self, header):
        self._header = header

    def get_data(self):
        ''' Lazy load of data '''
        if not self._data is None:
            return self._data
        cdf = self._header
        self._data = self._header.get_scaled_data()
        return self._data

    def get_shape(self):
        if not self._data is None:
            return self._data.shape
        return self._header.get_data_shape()
    
    def get_data_dtype(self):
        return self._header.get_data_dtype()
    
    @classmethod
    def from_filespec(klass, filespec):
        files = klass.filespec_to_files(filespec)
        return klass.from_files(files)
    
    @classmethod
    def from_files(klass, files):
        fname = files['image']
        header = klass._header_maker.from_fileobj(allopen(fname))
        affine = header.get_best_affine()
        ret =  klass(None, affine, header)
        ret._files = files
        return ret
    
    @classmethod
    def from_image(klass, img):
        return klass(img.get_data(),
                     img.get_affine(),
                     img.get_header(),
                     img.extra)
    
    @staticmethod
    def filespec_to_files(filespec):
        return {'image':filespec}
        
    @classmethod
    def load(klass, filespec):
        return klass.from_filespec(filespec)


load = MINCImage.load
