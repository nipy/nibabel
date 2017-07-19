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
import numpy as np

from .deprecated import deprecate_with_version
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

    The class ``__init__`` requires a spec which defines how the data will be
    read and rescaled. The spec may be a tuple of length 2 - 5, containing the
    shape, storage dtype, offset, slope and intercept, or a ``header`` object
    with methods:

    * get_data_shape
    * get_data_dtype
    * get_data_offset
    * get_slope_inter

    A header should also have a 'copy' method.  This requirement will go away
    when the deprecated 'header' propoerty goes away.

    This implementation allows us to deal with Analyze and its variants,
    including Nifti1, and with the MGH format.

    Other image types might need more specific classes to implement the API.
    See :mod:`nibabel.minc1`, :mod:`nibabel.ecat` and :mod:`nibabel.parrec` for
    examples.
    """
    # Assume Fortran array memory layout
    order = 'F'
    _header = None

    @kw_only_meth(2)
    def __init__(self, file_like, spec, mmap=True):
        """ Initialize array proxy instance

        Parameters
        ----------
        file_like : object
            File-like object or filename. If file-like object, should implement
            at least ``read`` and ``seek``.
        spec : object or tuple
            Tuple must have length 2-5, with the following values:

            #. shape: tuple - tuple of ints describing shape of data;
            #. storage_dtype: dtype specifier - dtype of array inside proxied
               file, or input to ``numpy.dtype`` to specify array dtype;
            #. offset: int - offset, in bytes, of data array from start of file
               (default: 0);
            #. slope: float - scaling factor for resulting data (default: 1.0);
            #. inter: float - intercept for rescaled data (default: 0.0).

            OR

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
        if hasattr(spec, 'get_data_shape'):
            slope, inter = spec.get_slope_inter()
            par = (spec.get_data_shape(),
                   spec.get_data_dtype(),
                   spec.get_data_offset(),
                   1. if slope is None else slope,
                   0. if inter is None else inter)
            # Reference to original header; we will remove this soon
            self._header = spec.copy()
        elif 2 <= len(spec) <= 5:
            optional = (0, 1., 0.)
            par = spec + optional[len(spec) - 2:]
        else:
            raise TypeError('spec must be tuple of length 2-5 or header object')

        # Copies of values needed to read array
        self._shape, self._dtype, self._offset, self._slope, self._inter = par
        # Permit any specifier that can be interpreted as a numpy dtype
        self._dtype = np.dtype(self._dtype)
        self._mmap = mmap

    @property
    @deprecate_with_version('ArrayProxy.header deprecated', '2.2', '3.0')
    def header(self):
        return self._header

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def offset(self):
        return self._offset

    @property
    def slope(self):
        return self._slope

    @property
    def inter(self):
        return self._inter

    @property
    def is_proxy(self):
        return True

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

    def reshape(self, shape):
        ''' Return an ArrayProxy with a new shape, without modifying data '''
        size = np.prod(self._shape)

        # Calculate new shape if not fully specified
        from operator import mul
        from functools import reduce
        n_unknowns = len([e for e in shape if e == -1])
        if n_unknowns > 1:
            raise ValueError("can only specify one unknown dimension")
        elif n_unknowns == 1:
            known_size = reduce(mul, shape, -1)
            unknown_size = size // known_size
            shape = tuple(unknown_size if e == -1 else e for e in shape)

        if np.prod(shape) != size:
            raise ValueError("cannot reshape array of size {:d} into shape "
                             "{!s}".format(size, shape))
        return self.__class__(file_like=self.file_like,
                              spec=(shape, self._dtype, self._offset,
                                    self._slope, self._inter),
                              mmap=self._mmap)


def is_proxy(obj):
    """ Return True if `obj` is an array proxy
    """
    try:
        return obj.is_proxy
    except AttributeError:
        return False


def reshape_dataobj(obj, shape):
    """ Use `obj` reshape method if possible, else numpy reshape function
    """
    return (obj.reshape(shape) if hasattr(obj, 'reshape')
            else np.reshape(obj, shape))
