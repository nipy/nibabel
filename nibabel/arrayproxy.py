# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Array proxy base class

The API is - at minimum:

* The object has a read-only attribute ``shape``
* read only ``is_proxy`` attribute / property
* the object returns the data array from ``np.asarray(obj)``
* that modifying no object outside ``obj`` will affect the result of
  ``np.asarray(obj)``.  Specifically:
  * Changes in position (``obj.tell()``) of passed file-like objects will
    not affect the output of from ``np.asarray(proxy)``.
  * if you pass a header into the __init__, then modifying the original
    header will not affect the result of the array return.
"""
import warnings

from .volumeutils import BinOpener, array_from_file, apply_read_scaling


class ArrayProxy(object):
    """
    The array proxy allows us to freeze the passed fileobj and header such that
    it returns the expected data array.

    This fairly generic implementation allows us to deal with Analyze and its
    variants, including Nifti1, and with the MGH format, apparently.

    It requires a ``header`` object with methods:
    * get_data_shape
    * get_data_dtype
    * get_data_offset
    * get_slope_inter

    The header should also have a 'copy' method.  This requirement will go away
    when the deprecated 'header' propoerty goes away.

    Other image types might need to implement their own implementation of this
    API.  See :mod:`minc` for an example.
    """
    def __init__(self, file_like, header):
        self.file_like = file_like
        # Copies of values needed to read array
        self._shape = header.get_data_shape()
        self._dtype = header.get_data_dtype()
        self._offset = header.get_data_offset()
        self._slope, self._inter = header.get_slope_inter()
        self._slope = 1.0 if self._slope is None else self._slope
        self._inter = 0.0 if self._inter is None else self._inter
        # Reference to original header; we will remove this soon
        self._header = header.copy()

    @property
    def header(self):
        warnings.warn('We will remove the header property from proxies soon',
                      FutureWarning,
                      stacklevel=2)
        return self._header

    @property
    def shape(self):
        return self._shape

    @property
    def is_proxy(self):
        return True

    @property
    def slope(self):
        return self._slope

    @property
    def inter(self):
        return self._inter

    @property
    def offset(self):
        return self._offset

    def __array__(self):
        ''' Read of data from file '''
        with BinOpener(self.file_like) as fileobj:
            raw_data = array_from_file(self._shape,
                                       self._dtype,
                                       fileobj,
                                       self._offset)
        # Upcast as necessary for big slopes, intercepts
        return apply_read_scaling(raw_data, self._slope, self._inter)
