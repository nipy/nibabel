""" Utilties for casting floats to integers
"""

import numpy as np

from .floating import floor_exact

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


class CastingError(Exception):
    pass


def nice_round(arr, int_type, nan2zero=True, infmax=False):
    """ Round floating point array `arr` to type `int_type`

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
        value, -np.inf as `int_type` integer minimum.  If False, merely set infs
        to be numbers at or near the maximum / minumum number in `arr` that can be
        contained in `int_type`.  Therefore False gives faster conversion at the
        expense of infs that are further from infinity.

    Returns
    -------
    iarr : ndarray
        of type `int_type`

    Examples
    --------
    >>> nice_round([np.nan, np.inf, -np.inf, 1.1, 6.6], np.int16)
    array([     0,  32767, -32768,      1,      7], dtype=int16)
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
