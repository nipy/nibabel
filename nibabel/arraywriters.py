""" Array writer objects

Array writers have init signature::

    def __init__(self, array, out_dtype=None)

and methods

* scaling_needed() - returns True if array requires scaling for write
*.finite_range() - returns min, max of self.array
* to_fileobj(fileobj, offset=None, order='F')

They have attributes:

* array
* out_dtype

They are designed to write arrays to a fileobj with reasonable memory
efficiency.

Array writers may be able to scale the array or apply an intercept, or do
something else to make sense of conversions between float and int, or between
larger ints and smaller.
"""

import numpy as np

from .casting import (int_to_float, as_int, int_abs, type_info, floor_exact,
                      best_float)
from .volumeutils import finite_range, array_to_file


class WriterError(Exception):
    pass


class ScalingError(WriterError):
    pass


class ArrayWriter(object):

    def __init__(self, array, out_dtype=None, calc_scale=True):
        """ Initialize array writer

        Parameters
        ----------
        array : array-like
            array-like object
        out_dtype : None or dtype
            dtype with which `array` will be written.  For this class,
            `out_dtype`` needs to be the same as the dtype of the input `array`
            or a swapped version of the same.
        \*\*kwargs : keyword arguments

        Examples
        --------
        >>> arr = np.array([0, 255], np.uint8)
        >>> aw = ArrayWriter(arr)
        >>> aw = ArrayWriter(arr, np.int8) #doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        WriterError: Scaling needed but cannot scale
        """
        self._array = np.asanyarray(array)
        arr_dtype = self._array.dtype
        if out_dtype is None:
            out_dtype = arr_dtype
        else:
            out_dtype = np.dtype(out_dtype)
        self._out_dtype = out_dtype
        self._finite_range = None
        if self.scaling_needed():
            raise WriterError("Scaling needed but cannot scale")

    def scaling_needed(self):
        """ Checks if scaling is needed for input array

        Raises WriterError if no scaling possible.

        The rules are in the code, but:
        * If numpy will cast, return False (no scaling needed)
        * If input or output is an object or structured type, raise
        * If input is complex, raise
        * If the output is float, return False
        * If there is no finite value in the input array, or the input array is
          all 0, return False (the writer will strip the non-finite values)
        * By now we are casting to (u)int. If the input type is a float, return
          True (we do need scaling)
        * Now input and output types are (u)ints. If the min and max in the data
          are within range of the output type, return False
        * Otherwise return True
        """
        data = self._array
        arr_dtype = data.dtype
        out_dtype = self._out_dtype
        # There's a bug in np.can_cast (at least up to and including 1.6.1) such
        # that any structured output type passes.  Check for this first.
        if 'V' in (arr_dtype.kind, out_dtype.kind):
            if arr_dtype == out_dtype:
                return False
            raise WriterError('Cannot cast to or from non-numeric types')
        if np.can_cast(arr_dtype, out_dtype):
            return False
        # Direct casting for complex output from any numeric type
        if out_dtype.kind == 'c':
            return False
        if arr_dtype.kind == 'c':
            raise WriterError('Cannot cast complex types to non-complex')
        # Direct casting for float output from any non-complex numeric type
        if out_dtype.kind == 'f':
            return False
        # Now we need to look at the data for special cases
        mn, mx = self.finite_range() # this is cached
        if (mn, mx) in ((0, 0), (np.inf, -np.inf)):
            # Data all zero, or no data is finite
            return False
        # Floats -> (u)ints always need scaling
        if arr_dtype.kind == 'f':
            return True
        # (u)int input, (u)int output
        assert arr_dtype.kind in 'iu' and out_dtype.kind in 'iu'
        info = np.iinfo(out_dtype)
        # No scaling needed if data already fits in output type
        # But note - we need to convert to ints, to avoid conversion to float
        # during comparisons, and therefore int -> float conversions which are
        # not exact.  Only a problem for uint64 though.  We need as_int here to
        # work around a numpy 1.4.1 bug in uint conversion
        if as_int(mn) >= as_int(info.min) and as_int(mx) <= as_int(info.max):
            return False
        return True

    @property
    def array(self):
        """ Return array from arraywriter """
        return self._array

    @property
    def out_dtype(self):
        """ Return `out_dtype` from arraywriter """
        return self._out_dtype

    def finite_range(self):
        """ Return (maybe cached) finite range of data array """
        if self._finite_range is None:
            self._finite_range = finite_range(self._array)
        return self._finite_range

    def _writing_range(self):
        """ Finite range for thresholding on write """
        if self._out_dtype.kind in 'iu' and self._array.dtype.kind == 'f':
            mn, mx = self.finite_range()
            if (mn, mx) == (np.inf, -np.inf): # no finite data
                mn, mx = 0, 0
            return mn, mx
        return None, None

    def to_fileobj(self, fileobj, order='F', nan2zero=True):
        """ Write array into `fileobj`

        Parameters
        ----------
        fileobj : file-like object
        order : {'F', 'C'}
            order (Fortran or C) to which to write array
        nan2zero : {True, False}, optional
            Whether to set NaN values to 0 when writing integer output.
            Defaults to True.  If False, NaNs get converted with numpy
            ``astype``, and the behavior is undefined.  Ignored for floating
            point output.
        """
        mn, mx = self._writing_range()
        array_to_file(self._array,
                      fileobj,
                      self._out_dtype,
                      offset=None,
                      mn=mn,
                      mx=mx,
                      order=order,
                      nan2zero=nan2zero)


class SlopeArrayWriter(ArrayWriter):
    """ ArrayWriter that can use scalefactor for writing arrays

    The scalefactor allows the array writer to write floats to int output types,
    and rescale larger ints to smaller.  It can therefore lose precision.

    It extends the ArrayWriter class with attribute:

    * slope

    and methods:

    * reset() - reset slope to default (not adapted to self.array)
    * calc_scale() - calculate slope to best write self.array
    """

    def __init__(self, array, out_dtype=None, calc_scale=True,
                 scaler_dtype=np.float32):
        """ Initialize array writer

        Parameters
        ----------
        array : array-like
            array-like object
        out_dtype : None or dtype
            dtype with which `array` will be written.  For this class,
            `out_dtype`` needs to be the same as the dtype of the input `array`
            or a swapped version of the same.
        calc_scale : {True, False}, optional
            Whether to calculate scaling for writing `array` on initialization.
            If False, then you can calculate this scaling with
            ``obj.calc_scale()`` - see examples
        scaler_dtype : dtype-like, optional
            specifier for numpy dtype for scaling

        Examples
        --------
        >>> arr = np.array([0, 254], np.uint8)
        >>> aw = SlopeArrayWriter(arr)
        >>> aw.slope
        1.0
        >>> aw = SlopeArrayWriter(arr, np.int8)
        >>> aw.slope
        2.0
        >>> aw = SlopeArrayWriter(arr, np.int8, calc_scale=False)
        >>> aw.slope
        1.0
        >>> aw.calc_scale()
        >>> aw.slope
        2.0
        """
        self._array = np.asanyarray(array)
        arr_dtype = self._array.dtype
        if out_dtype is None:
            out_dtype = arr_dtype
        else:
            out_dtype = np.dtype(out_dtype)
        self._out_dtype = out_dtype
        self.scaler_dtype = np.dtype(scaler_dtype)
        self.reset()
        if calc_scale:
            self.calc_scale()

    def reset(self):
        """ Set object to values before any scaling calculation """
        self.slope = 1.0
        self._finite_range = None
        self._scale_calced = False

    def _get_slope(self):
        return self._slope
    def _set_slope(self, val):
        self._slope = np.squeeze(self.scaler_dtype.type(val))
    slope = property(_get_slope, _set_slope, None, 'get/set slope')

    def calc_scale(self, force=False):
        """ Calculate / set scaling for floats/(u)ints to (u)ints
        """
        # If we've run already, return unless told otherwise
        if not force and self._scale_calced:
            return
        self.reset()
        if not self.scaling_needed():
            return
        self._do_scaling()
        self._scale_calced = True

    def to_fileobj(self, fileobj, order='F', nan2zero=True):
        """ Write array into `fileobj`

        Parameters
        ----------
        fileobj : file-like object
        order : {'F', 'C'}
            order (Fortran or C) to which to write array
        nan2zero : {True, False}, optional
            Whether to set NaN values to 0 when writing integer output.
            Defaults to True.  If False, NaNs get converted with numpy
            ``astype``, and the behavior is undefined.  Ignored for floating
            point output.
        """
        mn, mx = self._writing_range()
        array_to_file(self._array,
                      fileobj,
                      self._out_dtype,
                      offset=None,
                      divslope=self.slope,
                      mn=mn,
                      mx=mx,
                      order=order,
                      nan2zero=nan2zero)

    def _do_scaling(self):
        arr = self._array
        out_dtype = self._out_dtype
        assert out_dtype.kind in 'iu'
        mn, mx = self.finite_range()
        if arr.dtype.kind == 'f':
            # Float to (u)int scaling
            self._range_scale()
            return
        # (u)int to (u)int
        info = np.iinfo(out_dtype)
        out_max, out_min = info.max, info.min
        # If left as int64, uint64, comparisons will default to floats, and
        # these are inexact for > 2**53 - so convert to int
        if (as_int(mx) <= as_int(out_max) and
            as_int(mn) >= as_int(out_min)):
            # already in range
            return
        # (u)int to (u)int scaling
        self._iu2iu()

    def _iu2iu(self):
        # (u)int to (u)int scaling
        mn, mx = self.finite_range()
        if self._out_dtype.kind == 'u':
            # We're checking for a sign flip.  This can only work for uint
            # output, because, for int output, the abs min of the type is
            # greater than the abs max, so the data either fit into the range
            # (tested for in _do_scaling), or this test can't pass
            # Need abs that deals with max neg ints. abs problem only arises
            # when all the data is set to max neg integer value
            imax = np.iinfo(self._out_dtype).max
            if mx <= 0 and int_abs(mn) <= imax: # sign flip enough?
                # -1.0 * arr will be in scaler_dtype precision
                self.slope = -1.0
                return
        self._range_scale()

    def _range_scale(self):
        """ Calculate scaling based on data range and output type """
        mn, mx = self.finite_range() # These can be floats or integers
        out_dtype = self._out_dtype
        info = type_info(out_dtype)
        t_mn_mx = info['min'], info['max']
        big_float = best_float()
        if out_dtype.kind == 'f':
            # But we want maximum precision for the calculations. Casting will
            # not lose precision because min/max are of fp type.
            t_min, t_max = np.array(t_mn_mx, dtype = big_float)
        else: # (u)int
            t_min, t_max = [int_to_float(v, big_float) for v in t_mn_mx]
        if self._out_dtype.kind == 'u':
            if mn < 0 and mx > 0:
                raise WriterError('Cannot scale negative and positive '
                                  'numbers to uint without intercept')
            if mx <= 0: # All input numbers <= 0
                self.slope = mn / t_max
            else: # All input numbers > 0
                self.slope = mx / t_max
            return
        # Scaling to int. We need the bigger slope of (mn/t_min) and
        # (mx/t_max). If the mn or the max is the wrong side of 0, that
        # will make these negative and so they won't worry us
        mx_slope = mx / t_max
        mn_slope = mn / t_min
        self.slope = np.max([mx_slope, mn_slope])


class SlopeInterArrayWriter(SlopeArrayWriter):
    """ Array writer that can use slope and intercept to scale array

    The writer can subtract an intercept, and divided by a slope, in order to
    be able to convert floating point values into a (u)int range, or to convert
    larger (u)ints to smaller.

    It extends the ArrayWriter class with attributes:

    * inter
    * slope

    and methods:

    * reset() - reset inter, slope to default (not adapted to self.array)
    * calc_scale() - calculate inter, slope to best write self.array
    """

    def __init__(self, array, out_dtype=None, calc_scale=True,
                 scaler_dtype=np.float32):
        """ Initialize array writer

        Parameters
        ----------
        array : array-like
            array-like object
        out_dtype : None or dtype
            dtype with which `array` will be written.  For this class,
            `out_dtype`` needs to be the same as the dtype of the input `array`
            or a swapped version of the same.
        calc_scale : {True, False}, optional
            Whether to calculate scaling for writing `array` on initialization.
            If False, then you can calculate this scaling with
            ``obj.calc_scale()`` - see examples
        scaler_dtype : dtype-like, optional
            specifier for numpy dtype for slope, intercept

        Examples
        --------
        >>> arr = np.array([0, 255], np.uint8)
        >>> aw = SlopeInterArrayWriter(arr)
        >>> aw.slope, aw.inter
        (1.0, 0.0)
        >>> aw = SlopeInterArrayWriter(arr, np.int8)
        >>> (aw.slope, aw.inter) == (1.0, 128)
        True
        >>> aw = SlopeInterArrayWriter(arr, np.int8, calc_scale=False)
        >>> aw.slope, aw.inter
        (1.0, 0.0)
        >>> aw.calc_scale()
        >>> (aw.slope, aw.inter) == (1.0, 128)
        True
        """
        super(SlopeInterArrayWriter, self).__init__(array,
                                                    out_dtype,
                                                    calc_scale,
                                                    scaler_dtype)

    def reset(self):
        """ Set object to values before any scaling calculation """
        super(SlopeInterArrayWriter, self).reset()
        self.inter = 0.0

    def _get_inter(self):
        return self._inter
    def _set_inter(self, val):
        self._inter = np.squeeze(self.scaler_dtype.type(val))
    inter = property(_get_inter, _set_inter, None, 'get/set inter')

    def to_fileobj(self, fileobj, order='F', nan2zero=True):
        """ Write array into `fileobj`

        Parameters
        ----------
        fileobj : file-like object
        order : {'F', 'C'}
            order (Fortran or C) to which to write array
        nan2zero : {True, False}, optional
            Whether to set NaN values to 0 when writing integer output.
            Defaults to True.  If False, NaNs get converted with numpy
            ``astype``, and the behavior is undefined.  Ignored for floating
            point output.
        """
        mn, mx = self._writing_range()
        array_to_file(self._array,
                      fileobj,
                      self._out_dtype,
                      offset=None,
                      intercept=self.inter,
                      divslope=self.slope,
                      mn=mn,
                      mx=mx,
                      order=order,
                      nan2zero=nan2zero)

    def _iu2iu(self):
        # (u)int to (u)int
        mn, mx = [as_int(v) for v in self.finite_range()]
        # range may be greater than the largest integer for this type.
        # as_int needed to work round numpy 1.4.1 int casting bug
        out_dtype = self._out_dtype
        t_min, t_max = np.iinfo(out_dtype).min, np.iinfo(out_dtype).max
        type_range = as_int(t_max) - as_int(t_min)
        mn2mx = mx - mn
        if mn2mx <= type_range: # might offset be enough?
            if t_min == 0: # uint output - take min to 0
                # decrease offset with floor_exact, meaning mn >= t_min after
                # subtraction.  But we may have pushed the data over t_max,
                # which we check below
                inter = floor_exact(mn - t_min, self.scaler_dtype)
            else: # int output - take midpoint to 0
                # ceil below increases inter, pushing scale up to 0.5 towards
                # -inf, because ints have abs min == abs max + 1
                midpoint = mn + as_int(np.ceil(mn2mx / 2.0))
                # Floor exact decreases inter, so pulling scaled values more
                # positive. This may make mx - inter > t_max
                inter = floor_exact(midpoint, self.scaler_dtype)
            # Need to check still in range after floor_exact-ing
            int_inter = as_int(inter)
            assert mn - int_inter >= t_min
            if mx - int_inter <= t_max:
                self.inter = inter
                return
        # Try slope options (sign flip) and then range scaling
        super(SlopeInterArrayWriter, self)._iu2iu()

    def _range_scale(self):
        """ Calculate scaling, intercept based on data range and output type """
        mn, mx = self.finite_range() # Values of self.array.dtype type
        out_dtype = self._out_dtype
        if mx == mn: # Only one number in array
            self.inter = mn
            return
        # Straight mx-mn can overflow.
        big_float = best_float() # usually longdouble except in win 32
        if mn.dtype.kind == 'f': # Already floats
            # float64 and below cast correctly to longdouble.  Longdouble needs
            # no casting
            mn2mx = np.diff(np.array([mn, mx], dtype=big_float))
        else: # max possible (u)int range is 2**64-1 (int64, uint64)
            # int_to_float covers this range.  On windows longdouble is the same
            # as double so mn2mx will be 2**64 - thus overestimating slope
            # slightly.  Casting to int needed to allow mx-mn to be larger than
            # the largest (u)int value
            mn2mx = int_to_float(as_int(mx) - as_int(mn), big_float)
        if out_dtype.kind == 'f':
            # Type range, these are also floats
            info = type_info(out_dtype)
            t_mn_mx = info['min'], info['max']
        else:
            t_mn_mx = np.iinfo(out_dtype).min, np.iinfo(out_dtype).max
            t_mn_mx= [int_to_float(v, big_float) for v in t_mn_mx]
        # We want maximum precision for the calculations. Casting will
        # not lose precision because min/max are of fp type.
        assert [v.dtype.kind for v in t_mn_mx] == ['f', 'f']
        scaled_mn2mx = np.diff(np.array(t_mn_mx, dtype = big_float))
        slope = mn2mx / scaled_mn2mx
        self.inter = mn - t_mn_mx[0] * slope
        self.slope = slope
        if not np.all(np.isfinite([self.slope, self.inter])):
            raise ScalingError("Slope / inter not both finite")


def get_slope_inter(writer):
    """ Return slope, intercept from array writer object

    Parameters
    ----------
    writer : ArrayWriter instance

    Returns
    -------
    slope : scalar
        slope in `writer` or 1.0 if not present
    inter : scalar
        intercept in `writer` or 0.0 if not present

    Examples
    --------
    >>> arr = np.arange(10)
    >>> get_slope_inter(ArrayWriter(arr))
    (1.0, 0.0)
    >>> get_slope_inter(SlopeArrayWriter(arr))
    (1.0, 0.0)
    >>> get_slope_inter(SlopeInterArrayWriter(arr))
    (1.0, 0.0)
    """
    try:
        slope = writer.slope
    except AttributeError:
        slope = 1.0
    try:
        inter = writer.inter
    except AttributeError:
        inter = 0.0
    return slope, inter


def make_array_writer(data, out_type, has_slope=True, has_intercept=True,
                      **kwargs):
    """ Make array writer instance for array `data` and output type `out_type`

    Parameters
    ----------
    data : array-like
        array for which to create array writer
    out_type : dtype-like
        input to numpy dtype to specify array writer output type
    has_slope : {True, False}
        If True, array write can use scaling to adapt the array to `out_type`
    has_intercept : {True, False}
        If True, array write can use intercept to adapt the array to `out_type`
    \*\*kwargs : other keyword arguments
        to pass to the arraywriter class, if it accepts them.

    Returns
    -------
    writer : arraywriter instance
        Instance of array writer, with class adapted to `has_intercept` and
        `has_slope`.

    Examples
    --------
    >>> aw = make_array_writer(np.arange(10), np.uint8, True, True)
    >>> type(aw) == SlopeInterArrayWriter
    True
    >>> aw = make_array_writer(np.arange(10), np.uint8, True, False)
    >>> type(aw) == SlopeArrayWriter
    True
    >>> aw = make_array_writer(np.arange(10), np.uint8, False, False)
    >>> type(aw) == ArrayWriter
    True
    """
    data = np.asarray(data)
    if has_intercept == True and has_slope == False:
        raise ValueError('Cannot handle intercept without slope')
    if has_intercept:
        return SlopeInterArrayWriter(data, out_type, **kwargs)
    if has_slope:
        return SlopeArrayWriter(data, out_type, **kwargs)
    return ArrayWriter(data, out_type)
