""" Working with IEEE floating point values

Getting nearest exact integers in particular floating point types
"""
import numpy as np

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
    nmant = np.finfo(flt_type).nmant
    # Assuming the np.float type is always IEEE 64 bit
    if flt_type is np.float and nmant == 52:
        return 52
    # Now we should be testing long doubles
    assert flt_type is np.longdouble
    if nmant == 63: # 80-bit intel type
        return 63 # No including explicit first digit
    raise FloatingError('Cannot be confident of nmant value for %s' % flt_type)


def as_int(x, check=True):
    """ Return python integer representation of number

    This is useful because the numpy int(val) mechanism is broken for large
    values in np.longdouble

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
    """
    ip = 0
    rem = abs(x)
    while rem>=2:
        ip += 1
        rem /= 2
    return ip
