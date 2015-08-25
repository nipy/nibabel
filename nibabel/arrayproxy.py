# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Array proxy base class

The proxy API is - at minimum:

* The object has a read-only attribute ``shape``
* read only ``is_proxy`` attribute / property set to True
* the object returns the data array from ``np.asarray(prox)``
* returns array slice from ``prox[<slice_spec>]`` where ``<slice_spec>`` is any
  ndarray slice specification that does not use numpy 'advanced indexing'.
* modifying no object outside ``obj`` will affect the result of
  ``np.asarray(obj)``.  Specifically:

  * Changes in position (``obj.tell()``) of passed file-like objects will
    not affect the output of from ``np.asarray(proxy)``.
  * if you pass a header into the __init__, then modifying the original
    header will not affect the result of the array return.

See :mod:`nibabel.tests.test_proxy_api` for proxy API conformance checks.
"""
import warnings

from .volumeutils import array_from_file, apply_read_scaling
from .fileslice import fileslice
from .keywordonly import kw_only_meth
from .openers import ImageOpener


class ArrayProxy(object):
    """ Class to act as proxy for the array that can be read from a file

    The array proxy allows us to freeze the passed fileobj and header such that
    it returns the expected data array.

    This implementation assumes a contiguous array in the file object, with one
    of the numpy dtypes, starting at a given file position ``offset`` with
    single ``slope`` and ``intercept`` scaling to produce output values.

    The class ``__init__`` requires a ``header`` object with methods:

    * get_data_shape
    * get_data_dtype
    * get_data_offset
    * get_slope_inter

    The header should also have a 'copy' method.  This requirement will go away
    when the deprecated 'header' propoerty goes away.

    This implementation allows us to deal with Analyze and its variants,
    including Nifti1, and with the MGH format.

    Other image types might need more specific classes to implement the API.
    See :mod:`nibabel.minc1`, :mod:`nibabel.ecat` and :mod:`nibabel.parrec` for
    examples.
    """
    # Assume Fortran array memory layout
    order = 'F'

    @kw_only_meth(2)
    def __init__(self, file_like, header, mmap=True):
        """ Initialize array proxy instance

        Parameters
        ----------
        file_like : object
            File-like object or filename. If file-like object, should implement
            at least ``read`` and ``seek``.
        header : object
            Header object implementing ``get_data_shape``, ``get_data_dtype``,
            ``get_data_offset``, ``get_slope_inter``
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading data.
            If False, do not try numpy ``memmap`` for data array.  If one of
            {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A `mmap` value of
            True gives the same behavior as ``mmap='c'``.  If `file_like`
            cannot be memory-mapped, ignore `mmap` value and read array from
            file.
        scaling : {'fp', 'dv'}, optional, keyword only
            Type of scaling to use - see header ``get_data_scaling`` method.
        """
        if mmap not in (True, False, 'c', 'r'):
            raise ValueError("mmap should be one of {True, False, 'c', 'r'}")
        self.file_like = file_like
        # Copies of values needed to read array
        self._shape = header.get_data_shape()
        self._dtype = header.get_data_dtype()
        self._offset = header.get_data_offset()
        self._slope, self._inter = header.get_slope_inter()
        self._slope = 1.0 if self._slope is None else self._slope
        self._inter = 0.0 if self._inter is None else self._inter
        self._mmap = mmap
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

    def get_unscaled(self):
        ''' Read of data from file

        This is an optional part of the proxy API
        '''
        with ImageOpener(self.file_like) as fileobj:
            raw_data = array_from_file(self._shape,
                                       self._dtype,
                                       fileobj,
                                       offset=self._offset,
                                       order=self.order,
                                       mmap=self._mmap)
        return raw_data

    def __array__(self):
        # Read array and scale
        raw_data = self.get_unscaled()
        return apply_read_scaling(raw_data, self._slope, self._inter)

    def __getitem__(self, slicer):
        with ImageOpener(self.file_like) as fileobj:
            raw_data = fileslice(fileobj,
                                 slicer,
                                 self._shape,
                                 self._dtype,
                                 self._offset,
                                 order=self.order)
        # Upcast as necessary for big slopes, intercepts
        return apply_read_scaling(raw_data, self._slope, self._inter)


def is_proxy(obj):
    """ Return True if `obj` is an array proxy
    """
    try:
        return obj.is_proxy
    except AttributeError:
        return False
