''' Array proxy for analyze-type images '''

from nibabel.volumeutils import allopen
from nibabel.header_ufuncs import read_data


class AnalyzeArrayProxy(object):
    def __init__(self, file_like, header):
        self.file_like = file_like
        self.header = header.copy()
        self._data = None
        self.shape = header.get_data_shape()
        
    def __array__(self):
        if self._data is not None:
            return self._data
        fileobj = allopen(self.file_like)
        self._data = read_data(self.header, fileobj)
        return self._data
