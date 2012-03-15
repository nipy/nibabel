""" Utilties for casting floats to integers
"""

from platform import processor

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
    nan2zero : {True, False, None}
        Whether to convert NaN value to zero.  Default is True.  If False, and
        NaNs are present, raise CastingError. If None, do not check for NaN
        values and pass through directly to the ``astype`` casting mechanism.
        In this last case, the resulting value is undefined.
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
    mn, mx = shared_range(flt_type, int_type)
    if nan2zero is None:
        seen_nans = False
    else:
        nans = np.isnan(arr)
        seen_nans = np.any(nans)
        if nan2zero == False and seen_nans:
            raise CastingError('NaNs in array, nan2zero is False')
    iarr = np.clip(np.rint(arr), mn, mx).astype(int_type)
    if seen_nans:
        iarr[nans] = 0
    if not infmax:
        return iarr.reshape(shape)
    ii = np.iinfo(int_type)
    iarr[arr == np.inf] = ii.max
    if ii.min != int(mn):
        iarr[arr == -np.inf] = ii.min
    return iarr.reshape(shape)


# Cache range values
_SHARED_RANGES = {}

def shared_range(flt_type, int_type):
    """ Min and max in float type that are >=min, <=max in integer type

    This is not as easy as it sounds, because the float type may not be able to
    exactly represent the max or min integer values, so we have to find the next
    exactly representable floating point value to do the thresholding.

    Parameters
    ----------
    flt_type : dtype specifier
        A dtype specifier referring to a numpy floating point type.  For
        example, ``f4``, ``np.dtype('f4')``, ``np.float32`` are equivalent.
    int_type : dtype specifier
        A dtype specifier referring to a numpy integer type.  For example,
        ``i4``, ``np.dtype('i4')``, ``np.int32`` are equivalent

    Returns
    -------
    mn : object
        Number of type `flt_type` that is the minumum value in the range of
        `int_type`, such that ``mn.astype(int_type)`` >= min of `int_type`
    mx : object
        Number of type `flt_type` that is the maximum value in the range of
        `int_type`, such that ``mx.astype(int_type)`` <= max of `int_type`

    Examples
    --------
    >>> shared_range(np.float32, np.int32)
    (-2147483648.0, 2147483520.0)
    >>> shared_range('f4', 'i4')
    (-2147483648.0, 2147483520.0)
    """
    flt_type = np.dtype(flt_type).type
    int_type = np.dtype(int_type).type
    key = (flt_type, int_type)
    # Used cached value if present
    try:
        return _SHARED_RANGES[key]
    except KeyError:
        pass
    ii = np.iinfo(int_type)
    mn_mx = floor_exact(ii.min, flt_type), floor_exact(ii.max, flt_type)
    _SHARED_RANGES[key] = mn_mx
    return mn_mx

# ----------------------------------------------------------------------------
# Routines to work out the next lowest representable integer in floating point
# types.
# ----------------------------------------------------------------------------

try:
    _float16 = np.float16
except AttributeError: # float16 not present in np < 1.6
    _float16 = None


class FloatingError(Exception):
    pass


def type_info(np_type):
    """ Return dict with min, max, nexp, nmant, width for numpy type `np_type`

    Type can be integer in which case nexp and nmant are None.

    Parameters
    ----------
    np_type : numpy type specifier
        Any specifier for a numpy dtype

    Returns
    -------
    info : dict
        with fields ``min`` (minimum value), ``max`` (maximum value), ``nexp``
        (exponent width), ``nmant`` (significand precision not including
        implicit first digit) ``width`` (width in bytes). ``nexp``, ``nmant``
        are None for integer types. Both ``min`` and ``max`` are of type
        `np_type`.

    Raises
    ------
    FloatingError : for floating point types we don't recognize

    Notes
    -----
    You might be thinking that ``np.finfo`` does this job, and it does, except
    for PPC long doubles (http://projects.scipy.org/numpy/ticket/2077). This
    routine protects against errors in ``np.finfo`` by only accepting values
    that we know are likely to be correct.
    """
    dt = np.dtype(np_type)
    np_type = dt.type
    width = dt.itemsize
    try: # integer type
        info = np.iinfo(dt)
    except ValueError:
        pass
    else:
        return dict(min=np_type(info.min), max=np_type(info.max),
                    nmant=None, nexp=None, width=width)
    info = np.finfo(dt)
    # Trust the standard IEEE types
    nmant, nexp = info.nmant, info.nexp
    ret = dict(min=np_type(info.min), max=np_type(info.max), nmant=nmant,
               nexp=nexp, width=width)
    if np_type in (_float16, np.float32, np.float64,
                   np.complex64, np.complex128):
        return ret
    if dt.kind == 'c':
        assert np_type is np.longcomplex
        vals = (nmant, nexp, width / 2)
    else:
        assert np_type is np.longdouble
        vals = (nmant, nexp, width)
    if vals in ((112, 15, 16), # binary128
                (63, 15, 12), (63, 15, 16)): # Intel extended 80
        pass # these are OK
    elif vals == (1, 1, 16) and processor() == 'powerpc': # broken PPC
        dbl_info = np.finfo(np.float64)
        return dict(min=np_type(dbl_info.min), max=np_type(dbl_info.max),
                    nmant=106, nexp=11, width=width)
    else: # don't recognize the type
        raise FloatingError('We had not expected this type')
    return ret


def as_int(x, check=True):
    """ Return python integer representation of number

    This is useful because the numpy int(val) mechanism is broken for large
    values in np.longdouble.

    It is also useful to work around a numpy 1.4.1 bug in conversion of uints to
    python ints.

    This routine will still raise an OverflowError for values that are outside
    the range of float64.

    Parameters
    ----------
    x : object
        integer, unsigned integer or floating point value
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
    >>> as_int(2.1) #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    FloatingError: Not an integer: 2.1
    >>> as_int(2.1, check=False)
    2
    """
    x = np.array(x)
    if x.dtype.kind in 'iu':
        # This works around a nasty numpy 1.4.1 bug such that:
        # >>> int(np.uint32(2**32-1)
        # -1
        return int(str(x))
    ix = int(x)
    if ix == x:
        return ix
    fx = np.floor(x)
    if check and fx != x:
        raise FloatingError('Not an integer: %s' % x)
    if not fx.dtype.type == np.longdouble:
        return int(x)
    # Subtract float64 chunks until we have all of the number. If the int is too
    # large, it will overflow
    ret = 0
    while fx != 0:
        f64 = np.float64(fx)
        fx -= f64
        ret += int(f64)
    return ret


def int_to_float(val, flt_type):
    """ Convert integer `val` to floating point type `flt_type`

    Why is this so complicated?

    At least in numpy <= 1.6.1, numpy longdoubles do not correctly convert to
    ints, and ints do not correctly convert to longdoubles.  Specifically, in
    both cases, the values seem to go through float64 conversion on the way, so
    to convert better, we need to split into float64s and sum up the result.

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
    faval = np.longdouble(0)
    while val != 0:
        f64 = np.float64(val)
        faval += f64
        val -= int(f64)
    return faval


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
    try: # int_to_float deals with longdouble safely
        faval = int_to_float(aval, flt_type)
    except OverflowError:
        faval = np.inf
    info = type_info(flt_type)
    if faval == np.inf:
        return sign * info['max']
    if as_int(faval) <= aval: # as_int deals with longdouble safely
        # Float casting has made the value go down or stay the same
        return sign * faval
    # Float casting made the value go up
    biggest_gap = 2**(floor_log2(aval) - info['nmant'])
    assert biggest_gap > 1
    faval -= flt_type(biggest_gap)
    return sign * faval


def int_abs(arr):
    """ Absolute values of array taking care of max negative int values

    Parameters
    ----------
    arr : array-like

    Returns
    -------
    abs_arr : array
        array the same shape as `arr` in which all negative numbers have been
        changed to positive numbers with the magnitude.

    Examples
    --------
    This kind of thing is confusing in base numpy:

    >>> import numpy as np
    >>> np.abs(np.int8(-128))
    -128

    ``int_abs`` fixes that:

    >>> int_abs(np.int8(-128))
    128
    >>> int_abs(np.array([-128, 127], dtype=np.int8))
    array([128, 127], dtype=uint8)
    >>> int_abs(np.array([-128, 127], dtype=np.float32))
    array([ 128.,  127.], dtype=float32)
    """
    arr = np.array(arr, copy=False)
    dt = arr.dtype
    if dt.kind == 'u':
        return arr
    if dt.kind != 'i':
        return np.absolute(arr)
    out = arr.astype(np.dtype(dt.str.replace('i', 'u')))
    return np.choose(arr < 0, (arr, arr * -1), out=out)


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
        rem //= 2
    return ip
