''' Array proxy base class '''

class ArrayProxy(object):
    def __init__(self, file_like, header):
        self.file_like = file_like
        self.header = header.copy()
        self._data = None
        self.shape = header.get_data_shape()
        
    def __array__(self):
        ''' Cached read of data from file '''
        if self._data is None:
            self._data = self._read_data()
        return self._data

    def _read_data(self):
        raise NotImplementedError


