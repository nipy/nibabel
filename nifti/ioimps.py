''' IO implementatations '''

def guessed_imp(filespec):
    return IOImplementation.from_filespec(filespec)

class IOImplementation(object):
    def __init__(self, filespec = None):
        self._filespec = None
        self.set_filespec(filespec)
        
    def set_filespec(self, filespec):
        self._filespec = filespec

    def get_filespec(self):
        return self._filespec

    def to_filespec(self, filespec=None):
        raise ImplementationError

    def copy(self):
        raise ImplementationError

    def get_affine(self):
        raise ImplementationError

    def get_output_space(self):
        raise ImplementationError

    def set_data_shape(self, shape):
        raise ImplementationError

    def get_data_dtype(self):
        raise NotImplementedError

    def set_data_dtype(self, dtype):
        raise NotImplementedError

    def write_slice(data, slicedef, outfile = None):
        raise ImplementationError

default_io = IOImplementation

