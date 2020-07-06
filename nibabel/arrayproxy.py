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
from contextlib import contextmanager
from threading import RLock

import numpy as np

from .deprecated import deprecate_with_version
from .volumeutils import array_from_file, apply_read_scaling
from .fileslice import fileslice, canonical_slicers
from . import openers


"""This flag controls whether a new file handle is created every time an image
is accessed through an ``ArrayProxy``, or a single file handle is created and
used for the lifetime of the ``ArrayProxy``. It should be set to one of
``True`` or ``False``.

Management of file handles will be performed either by ``ArrayProxy`` objects,
or by the ``indexed_gzip`` package if it is used.

If this flag is set to ``True``, a single file handle is created and used. If
``False``, a new file handle is created every time the image is accessed.

If this is set to any other value, attempts to create an ``ArrayProxy`` without
specifying the ``keep_file_open`` flag will result in a ``ValueError`` being
raised.

.. warning:: Setting this flag to a value of ``'auto'`` became deprecated
             behaviour in version 2.4.1. Support for ``'auto'`` was removed
             in version 3.0.0.
"""
KEEP_FILE_OPEN_DEFAULT = False


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
    when the deprecated 'header' property goes away.

    This implementation allows us to deal with Analyze and its variants,
    including Nifti1, and with the MGH format.

    Other image types might need more specific classes to implement the API.
    See :mod:`nibabel.minc1`, :mod:`nibabel.ecat` and :mod:`nibabel.parrec` for
    examples.
    """
    # Assume Fortran array memory layout
    order = 'F'
    _header = None

    def __init__(self, file_like, spec, *, mmap=True, keep_file_open=None):
        """Initialize array proxy instance

        .. deprecated:: 2.4.1
            ``keep_file_open='auto'`` is redundant with `False` and has
            been deprecated. It raises an error as of nibabel 3.0.

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
        keep_file_open : { None, True, False }, optional, keyword only
            `keep_file_open` controls whether a new file handle is created
            every time the image is accessed, or a single file handle is
            created and used for the lifetime of this ``ArrayProxy``. If
            ``True``, a single file handle is created and used. If ``False``,
            a new file handle is created every time the image is accessed.
            If ``file_like`` is an open file handle, this setting has no
            effect. The default value (``None``) will result in the value of
            ``KEEP_FILE_OPEN_DEFAULT`` being used.
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
        # Flags to keep track of whether a single ImageOpener is created, and
        # whether a single underlying file handle is created.
        self._keep_file_open, self._persist_opener = \
            self._should_keep_file_open(file_like, keep_file_open)
        self._lock = RLock()

    def __del__(self):
        """If this ``ArrayProxy`` was created with ``keep_file_open=True``,
        the open file object is closed if necessary.
        """
        if hasattr(self, '_opener') and not self._opener.closed:
            self._opener.close_if_mine()
            self._opener = None

    def __getstate__(self):
        """Returns the state of this ``ArrayProxy`` during pickling. """
        state = self.__dict__.copy()
        state.pop('_lock', None)
        return state

    def __setstate__(self, state):
        """Sets the state of this ``ArrayProxy`` during unpickling. """
        self.__dict__.update(state)
        self._lock = RLock()

    def _should_keep_file_open(self, file_like, keep_file_open):
        """Called by ``__init__``.

        This method determines how to manage ``ImageOpener`` instances,
        and the underlying file handles - the behaviour depends on:

         - whether ``file_like`` is an an open file handle, or a path to a
           ``'.gz'`` file, or a path to a non-gzip file.
         - whether ``indexed_gzip`` is present (see
           :attr:`.openers.HAVE_INDEXED_GZIP`).

        An ``ArrayProxy`` object uses two internal flags to manage
        ``ImageOpener`` instances and underlying file handles.

          - The ``_persist_opener`` flag controls whether a single
            ``ImageOpener`` should be created and used for the lifetime of
            this ``ArrayProxy``, or whether separate ``ImageOpener`` instances
            should be created on each file access.

          - The ``_keep_file_open`` flag controls qwhether the underlying file
            handle should be kept open for the lifetime of this
            ``ArrayProxy``, or whether the file handle should be (re-)opened
            and closed on each file access.

        The internal ``_keep_file_open`` flag is only relevant if
        ``file_like`` is a ``'.gz'`` file, and the ``indexed_gzip`` library is
        present.

        This method returns the values to be used for the internal
        ``_persist_opener`` and ``_keep_file_open`` flags; these values are
        derived according to the following rules:

        1. If ``file_like`` is a file(-like) object, both flags are set to
        ``False``.

        2. If ``keep_file_open`` (as passed to :meth:``__init__``) is
           ``True``, both internal flags are set to ``True``.

        3. If ``keep_file_open`` is ``False``, but ``file_like`` is not a path
           to a ``.gz`` file or ``indexed_gzip`` is not present, both flags
           are set to ``False``.

        4. If ``keep_file_open`` is ``False``, ``file_like`` is a path to a
           ``.gz`` file, and ``indexed_gzip`` is present, ``_persist_opener``
           is set to ``True``, and ``_keep_file_open`` is set to ``False``.
           In this case, file handle management is delegated to the
           ``indexed_gzip`` library.

        .. deprecated:: 2.4.1
            ``keep_file_open='auto'`` is redundant with `False` and has
            been deprecated. It raises an error as of nibabel 3.0.

        Parameters
        ----------

        file_like : object
            File-like object or filename, as passed to ``__init__``.
        keep_file_open : { True, False }
            Flag as passed to ``__init__``.

        Returns
        -------

        A tuple containing:
          - ``keep_file_open`` flag to control persistence of file handles
          - ``persist_opener`` flag to control persistence of ``ImageOpener``
            objects.
        """
        if keep_file_open is None:
            keep_file_open = KEEP_FILE_OPEN_DEFAULT
            if keep_file_open not in (True, False):
                raise ValueError("nibabel.arrayproxy.KEEP_FILE_OPEN_DEFAULT "
                                 f"must be boolean. Found: {keep_file_open}")
        elif keep_file_open not in (True, False):
            raise ValueError('keep_file_open must be one of {None, True, False}')

        # file_like is a handle - keep_file_open is irrelevant
        if hasattr(file_like, 'read') and hasattr(file_like, 'seek'):
            return False, False
        # if the file is a gzip file, and we have_indexed_gzip,
        have_igzip = openers.HAVE_INDEXED_GZIP and file_like.endswith('.gz')

        persist_opener = keep_file_open or have_igzip
        return keep_file_open, persist_opener

    @property
    @deprecate_with_version('ArrayProxy.header deprecated', '2.2', '3.0')
    def header(self):
        return self._header

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

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

    @contextmanager
    def _get_fileobj(self):
        """Create and return a new ``ImageOpener``, or return an existing one.

        The specific behaviour depends on the value of the ``keep_file_open``
        flag that was passed to ``__init__``.

        Yields
        ------
        ImageOpener
            A newly created ``ImageOpener`` instance, or an existing one,
            which provides access to the file.
        """
        if self._persist_opener:
            if not hasattr(self, '_opener'):
                self._opener = openers.ImageOpener(
                    self.file_like, keep_open=self._keep_file_open)
            yield self._opener
        else:
            with openers.ImageOpener(
                    self.file_like, keep_open=False) as opener:
                yield opener

    def _get_unscaled(self, slicer):
        if canonical_slicers(slicer, self._shape, False) == \
                canonical_slicers((), self._shape, False):
            with self._get_fileobj() as fileobj, self._lock:
                return array_from_file(self._shape,
                                       self._dtype,
                                       fileobj,
                                       offset=self._offset,
                                       order=self.order,
                                       mmap=self._mmap)
        with self._get_fileobj() as fileobj:
            return fileslice(fileobj,
                             slicer,
                             self._shape,
                             self._dtype,
                             self._offset,
                             order=self.order,
                             lock=self._lock)

    def _get_scaled(self, dtype, slicer):
        # Ensure scale factors have dtypes
        scl_slope = np.asanyarray(self._slope)
        scl_inter = np.asanyarray(self._inter)
        use_dtype = scl_slope.dtype if dtype is None else dtype

        if np.can_cast(scl_slope, use_dtype):
            scl_slope = scl_slope.astype(use_dtype)
        if np.can_cast(scl_inter, use_dtype):
            scl_inter = scl_inter.astype(use_dtype)
        # Read array and upcast as necessary for big slopes, intercepts
        scaled = apply_read_scaling(self._get_unscaled(slicer=slicer), scl_slope, scl_inter)
        if dtype is not None:
            scaled = scaled.astype(np.promote_types(scaled.dtype, dtype), copy=False)
        return scaled

    def get_unscaled(self):
        """ Read data from file

        This is an optional part of the proxy API
        """
        return self._get_unscaled(slicer=())

    def __array__(self, dtype=None):
        """ Read data from file and apply scaling, casting to ``dtype``

        If ``dtype`` is unspecified, the dtype of the returned array is the
        narrowest dtype that can represent the data without overflow.
        Generally, it is the wider of the dtypes of the slopes or intercepts.

        The types of the scale factors will generally be determined by the
        parameter size in the image header, and so should be consistent for a
        given image format, but may vary across formats.

        Parameters
        ----------
        dtype : numpy dtype specifier, optional
            A numpy dtype specifier specifying the type of the returned array.

        Returns
        -------
        array
            Scaled image data with type `dtype`.
        """
        arr = self._get_scaled(dtype=dtype, slicer=())
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def __getitem__(self, slicer):
        return self._get_scaled(dtype=None, slicer=slicer)

    def reshape(self, shape):
        """ Return an ArrayProxy with a new shape, without modifying data """
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
            raise ValueError(f"cannot reshape array of size {size:d} into shape {shape!s}")
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
