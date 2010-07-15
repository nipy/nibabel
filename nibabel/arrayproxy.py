# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
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


