''' Very simple spatial image class '''

class SpatialImage(object):
    _header_maker = dict
    ''' Template class for lightweight image '''
    def __init__(self, data, affine, header=None, extra=None):
        self._data = data
        self._affine = affine
        if extra is None:
            extra = {}
        self.extra = extra
        if header is None:
            self._header = self._header_maker()
        else:
            self._header = self._header_maker(endianness=header.endianness)
            for key, value in header.items():
                if key in self._header:
                    self._header[key] = value
                else:
                    self.extra[key] = value
        self._files = {}
        
    def __str__(self):
        shape = self.get_shape()
        affine = self.get_affine()
        return '\n'.join((
                str(self.__class__),
                'data shape %s' % (shape,),
                'affine: ',
                '%s' % affine,
                'metadata:',
                '%s' % self._header))

    def get_data(self):
        return self._data

    def get_shape(self):
        if self._data:
            return self._data.shape

    def get_data_dtype(self):
        raise NotImplementedError

    def set_data_dtype(self, dtype):
        raise NotImplementedError

    def get_affine(self):
        return self._affine

    def get_header(self):
        return self._header

    @classmethod
    def from_filespec(klass, filespec):
        raise NotImplementedError

    def from_files(klass, files):
        raise NotImplementedError

    def from_image(klass, img):
        raise NotImplementedError

    @staticmethod
    def filespec_to_files(filespec):
        raise NotImplementedError
    
    def to_filespec(self, filespec):
        raise NotImplementedError

    def to_files(self, files=None):
        raise NotImplementedError
