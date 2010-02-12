''' Array proxy for analyze-type images '''

from nibabel.volumeutils import allopen
from nibabel.header_ufuncs import read_data


class AnalyzeArrayProxy(object):
    def __init__(self, file_like, hdr):
        self.file_like = file_like
        self.hdr = hdr.copy()
        self._data = None
        
    def __array__(self):
        if self._data is not None:
            return self._data
        fileobj = allopen(self.file_like)
        self._data = read_data(self.hdr, fileobj)
        return self._data
