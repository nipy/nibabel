""" Utilties for casting floats to integers
"""

import numpy as np


class CastingError(Exception):
    pass


def float_to_int(arr, int_type, nan2zero=True, infmax=False):
    """ Convert floating point array `arr` to type `int_type`

    * Rounds numbers to nearest integer
    * Clips values to prevent overflows when casting
    * Converts NaN to 0 (for `nan2zero`==True

    Casting floats to integers is delicate because the result is undefined
    and platform specific for float values outside the range of `int_type`.
    Define ``shared_min`` to be the minimum value that can be exactly
    represented in both the float type of `arr` and `int_type`. Define
    `shared_max` to be the equivalent maximum value.  To avoid undefined results
    we threshold `arr` at ``shared_min`` and ``shared_max``.

    Parameters
    ----------
    arr : array-like
        Array of floating point type
    int_type : object
        Numpy integer type
    nan2zero : {True, False}
        Whether to convert NaN value to zero.  Default is True.  If False, and
        NaNs are present, raise CastingError
    infmax : {False, True}
        If True, set np.inf values in `arr` to be `int_type` integer maximum
        value, -np.inf as `int_type` integer minimum.  If False, set +/- infs to
        be ``shared_min``, ``shared_max`` as defined above.  Therefore False
        gives faster conversion at the expense of infs that are further from
        infinity.

    Returns
    -------
    iarr : ndarray
        of type `int_type`

    Examples
    --------
    >>> float_to_int([np.nan, np.inf, -np.inf, 1.1, 6.6], np.int16)
    array([     0,  32767, -32768,      1,      7], dtype=int16)

    Notes
    -----
    Numpy relies on the C library to cast from float to int using the standard
    ``astype`` method of the array.

    Quoting from section F4 of the C99 standard:

        If the floating value is infinite or NaN or if the integral part of the
        floating value exceeds the range of the integer type, then the
        "invalid" floating-point exception is raised and the resulting value
        is unspecified.

    Hence we threshold at ``shared_min`` and ``shared_max`` to avoid casting to
    values that are undefined.

    See: http://en.wikipedia.org/wiki/C99 . There are links to the C99 standard
    from that page.
    """
    arr = np.asarray(arr)
    flt_type = arr.dtype.type
    int_type = np.dtype(int_type).type
    # Deal with scalar as input; fancy indexing needs 1D
    shape = arr.shape
    arr = np.atleast_1d(arr)
    mn, mx = _cached_int_clippers(flt_type, int_type)
    nans = np.isnan(arr)
    have_nans = np.any(nans)
    if not nan2zero and have_nans:
        raise CastingError('NaNs in array, nan2zero not True')
    iarr = np.clip(np.rint(arr), mn, mx).astype(int_type)
    if have_nans:
        iarr[nans] = 0
    if not infmax:
        return iarr.reshape(shape)
    ii = np.iinfo(int_type)
    iarr[arr == np.inf] = ii.max
    if ii.min != int(mn):
        iarr[arr == -np.inf] = ii.min
    return iarr.reshape(shape)


def int_clippers(flt_type, int_type):
    """ Min and max in float type that are >=min, <=max in integer type

    This is not as easy as it sounds, because the float type may not be able to
    exactly represent the max or min integer values, so we have to find the next
    exactly representable floating point value to do the thresholding.

    Parameters
    ----------
    flt_type : object
        numpy floating point type
    int_type : object
        numpy integer type

    Returns
    -------
    mn : object
        Number of type `flt_type` that is the minumum value in the range of
        `int_type`, such that ``mn.astype(int_type)`` >= min of `int_type`
    mx : object
        Number of type `flt_type` that is the maximum value in the range of
        `int_type`, such that ``mx.astype(int_type)`` <= max of `int_type`
    """
    ii = np.iinfo(int_type)
    return floor_exact(ii.min, flt_type), floor_exact(ii.max, flt_type)


# Cache clip values
FLT_INT_CLIPS = {}

def _cached_int_clippers(flt_type, int_type):
    if not (flt_type, int_type) in FLT_INT_CLIPS:
        FLT_INT_CLIPS[flt_type, int_type] = int_clippers(flt_type, int_type)
    return FLT_INT_CLIPS[(flt_type, int_type)]

# ---------------------------------------------------------------------------
# Routines to work out the next lowest representable intger in floating point
# types.
# ---------------------------------------------------------------------------

try:
    _float16 = np.float16
except AttributeError: # float16 not present in np < 1.6
    _float16 = None

# The number of significand digits in IEEE floating point formats, not including
# the implicit leading 0.  See http://en.wikipedia.org/wiki/IEEE_754-2008
_flt_nmant = {
    _float16: 10,
    np.float32: 23,
    np.float64: 52,
    }


class FloatingError(Exception):
    pass


def flt2nmant(flt_type):
    """ Number of significand bits in float type `flt_type`

    Parameters
    ----------
    flt_type : object
        Numpy floating point type, such as np.float32

    Returns
    -------
    nmant : int
        Number of digits in the signficand
    """
    try:
        return _flt_nmant[flt_type]
    except KeyError:
        pass
    fi = np.finfo(flt_type)
    nmant, nexp = fi.nmant, fi.nexp
    # Assuming the np.float type is always IEEE 64 bit
    if flt_type is np.float and (nmant, nexp) == (52, 11):
        return 52
    # Now we should be testing long doubles
    assert flt_type is np.longdouble
    if (nmant, nexp) == (63, 15): # 80-bit intel type
        return 63 # Not including explicit first digit
    # We test the declared nmant by stepping up and down.  These tests assume a
    # binary format
    i_end_contig = 2**(nmant+1) # int
    f_end_contig = flt_type(i_end_contig)
    # We need as_int here because long doubles do not necessarily convert
    # correctly to ints with int() - see
    # http://projects.scipy.org/numpy/ticket/1395
    if as_int(f_end_contig-1) == (i_end_contig-1): # still representable
        if as_int(f_end_contig+1) == i_end_contig: # Rounding down
            return nmant
    raise FloatingError('Cannot be confident of nmant value for %s' % flt_type)


def as_int(x, check=True):
    """ Return python integer representation of number

    This is useful because the numpy int(val) mechanism is broken for large
    values in np.longdouble.

    This routine will still break for values that are outside the range of
    float64.

    Parameters
    ----------
    x : object
        Floating point value
    check : {True, False}
        If True, raise error for values that are not integers

    Returns
    -------
    i : int
        Python integer

    Examples
    --------
    >>> as_int(2.0)
    2
    >>> as_int(-2.0)
    -2
    >>> as_int(2.1)
    Traceback (most recent call last):
        ...
    FloatingError: Not an integer: 2.1
    >>> as_int(2.1, check=False)
    2
    """
    ix = int(x)
    if ix == x:
        return ix
    fx = np.floor(x)
    if check and fx != x:
        raise FloatingError('Not an integer: %s' % x)
    f64 = np.float64(fx)
    i64 = int(f64)
    assert f64 == i64
    res = fx - f64
    return ix + int(res)


def int_to_float(val, flt_type):
    """ Convert integer `val` to floating point type `flt_type`

    Useful because casting to ``np.longdouble`` loses precision as it appears to
    go through casting to np.float64.

    Parameters
    ----------
    val : int
        Integer value
    flt_type : object
        numpy floating point type

    Returns
    -------
    f : numpy scalar
        of type `flt_type`
    """
    if not flt_type is np.longdouble:
        return flt_type(val)
    f64 = np.float64(val)
    res = val - int(f64)
    return np.longdouble(f64) + np.longdouble(res)


def floor_exact(val, flt_type):
    """ Get nearest exact integer to `val`, towards 0, in float type `flt_type`

    Parameters
    ----------
    val : int
        We have to pass val as an int rather than the floating point type
        because large integers cast as floating point may be rounded by the
        casting process.
    flt_type : numpy type
        numpy float type.  Only IEEE types supported (np.float16, np.float32,
        np.float64)

    Returns
    -------
    floor_val : object
        value of same floating point type as `val`, that is the next excat
        integer in this type, towards zero, or == `val` if val is exactly
        representable.

    Examples
    --------
    Obviously 2 is within the range of representable integers for float32

    >>> floor_exact(2, np.float32)
    2.0

    As is 2**24-1 (the number of significand digits is 23 + 1 implicit)

    >>> floor_exact(2**24-1, np.float32) == 2**24-1
    True

    But 2**24+1 gives a number that float32 can't represent exactly

    >>> floor_exact(2**24+1, np.float32) == 2**24
    True
    """
    val = int(val)
    flt_type = np.dtype(flt_type).type
    sign = val > 0 and 1 or -1
    aval = abs(val)
    if flt_type is np.longdouble:
        # longdouble seems to go through casting to float64, so getting the
        # value into float128 with the given precision needs to go through two
        # steps, first float64, then adding the remainder.
        f64 = floor_exact(aval, np.float64)
        i64 = int(f64)
        assert f64 == i64
        res = aval - i64
        try:
            faval = flt_type(i64) + flt_type(res)
        except OverflowError:
            faval = np.inf
        if faval == np.inf:
            return sign * np.finfo(flt_type).max
        if (faval - f64) <= res:
            # Float casting has made the value go down or stay the same
            return sign * faval
    else: # Normal case
        try:
            faval = flt_type(aval)
        except OverflowError:
            faval = np.inf
        if faval == np.inf:
            return sign * np.finfo(flt_type).max
        if int(faval) <= aval:
            # Float casting has made the value go down or stay the same
            return sign * faval
    # Float casting made the value go up
    nmant = flt2nmant(flt_type)
    biggest_gap = 2**(floor_log2(aval) - nmant)
    assert biggest_gap > 1
    faval -= flt_type(biggest_gap)
    return sign * faval


def floor_log2(x):
    """ floor of log2 of abs(`x`)

    Embarrassingly, from http://en.wikipedia.org/wiki/Binary_logarithm

    Parameters
    ----------
    x : int

    Returns
    -------
    L : int
        floor of base 2 log of `x`

    Examples
    --------
    >>> floor_log2(2**9+1)
    9
    >>> floor_log2(-2**9+1)
    8
    """
    ip = 0
    rem = abs(x)
    while rem>=2:
        ip += 1
        rem /= 2
    return ip
