""" Test numerical errors introduced by writing then reading images

Test arrays with a range of numerical values, integer and floating point.
"""

import numpy as np

from ..externals.six import BytesIO
from .. import Nifti1Image
from ..spatialimages import HeaderDataError
from ..arraywriters import ScalingError
from ..casting import best_float, ulp, type_info

from nose.tools import assert_true

from numpy.testing import assert_array_equal, assert_almost_equal

DEBUG = True

def round_trip(arr, out_dtype):
    img = Nifti1Image(arr, np.eye(4))
    img.file_map['image'].fileobj = BytesIO()
    img.set_data_dtype(out_dtype)
    img.to_file_map()
    back = Nifti1Image.from_file_map(img.file_map)
    hdr = back.get_header()
    return (back.get_data(),) + hdr.get_slope_inter()


def check_params(in_arr, in_type, out_type):
    arr = in_arr.astype(in_type)
    # clip infs that can arise from downcasting
    if arr.dtype.kind == 'f':
        info = np.finfo(in_type)
        arr = np.clip(arr, info.min, info.max)
    try:
        arr_dash, slope, inter = round_trip(arr, out_type)
    except (ScalingError, HeaderDataError):
        return arr, None, None, None
    return arr, arr_dash, slope, inter


BFT = best_float()
LOGe2 = np.log(BFT(2))


def big_bad_ulp(arr):
    """ Return array of ulp values for values in `arr`

    I haven't thought about whether the vectorized log2 here could lead to
    incorrect rounding; this only needs to be ballpark

    This function might be used in nipy/io/tests/test_image_io.py

    Parameters
    ----------
    arr : array
        floating point array

    Returns
    -------
    ulps : array
        ulp values for each element of arr
    """
    # Assumes array is floating point
    arr = np.asarray(arr)
    info = type_info(arr.dtype)
    working_arr = np.abs(arr.astype(BFT))
    # Log2 for numpy < 1.3
    fl2 = np.zeros_like(working_arr) + info['minexp']
    # Avoid divide by zero error for log of 0
    nzs = working_arr > 0
    fl2[nzs] = np.floor(np.log(working_arr[nzs]) / LOGe2)
    fl2 = np.clip(fl2, info['minexp'], np.inf)
    return 2**(fl2 - info['nmant'])


def test_big_bad_ulp():
    for ftype in (np.float32, np.float64):
        ti = type_info(ftype)
        fi = np.finfo(ftype)
        min_ulp = 2 ** (ti['minexp'] - ti['nmant'])
        in_arr = np.zeros((10,), dtype=ftype)
        in_arr = np.array([0, 0, 1, 2, 4, 5, -5, -np.inf, np.inf], dtype=ftype)
        out_arr = [min_ulp, min_ulp, fi.eps, fi.eps * 2, fi.eps * 4,
                   fi.eps * 4, fi.eps * 4, np.inf, np.inf]
        assert_array_equal(big_bad_ulp(in_arr).astype(ftype), out_arr)


BIG_FLOAT = np.float64

def test_round_trip():
    scaling_type = np.float32
    rng = np.random.RandomState(20111121)
    N = 10000
    sd_10s = range(-20, 51, 5)
    iuint_types = np.sctypes['int'] + np.sctypes['uint']
    # Remove intp types, which cannot be set into nifti header datatype
    iuint_types.remove(np.intp)
    iuint_types.remove(np.uintp)
    f_types = [np.float32, np.float64]
    # Expanding standard deviations
    for i, sd_10 in enumerate(sd_10s):
        sd = 10.0**sd_10
        V_in = rng.normal(0, sd, size=(N,1))
        for j, in_type in enumerate(f_types):
            for k, out_type in enumerate(iuint_types):
                check_arr(sd_10, V_in, in_type, out_type, scaling_type)
    # Spread integers across range
    for i, sd in enumerate(np.linspace(0.05, 0.5, 5)):
        for j, in_type in enumerate(iuint_types):
            info = np.iinfo(in_type)
            mn, mx = info.min, info.max
            type_range = mx - mn
            center = type_range / 2.0 + mn
            # float(sd) because type_range can be type 'long'
            width = type_range * float(sd)
            V_in = rng.normal(center, width, size=(N,1))
            for k, out_type in enumerate(iuint_types):
                check_arr(sd, V_in, in_type, out_type, scaling_type)


def check_arr(test_id, V_in, in_type, out_type, scaling_type):
    arr, arr_dash, slope, inter = check_params(V_in, in_type, out_type)
    if arr_dash is None:
        return
    nzs = arr != 0 # avoid divide by zero error
    if not np.any(nzs):
        if DEBUG:
            raise ValueError('Array all zero')
        return
    arr = arr[nzs]
    arr_dash_L = arr_dash.astype(BIG_FLOAT)[nzs]
    top = arr - arr_dash_L
    if not np.any(top != 0):
        return
    rel_err = np.abs(top / arr)
    abs_err = np.abs(top)
    if slope == 1: # integers output, offset only scaling
        if (set((in_type, out_type)) == set((np.int64, np.uint64)) and
            type_info(BFT)['nmant'] < 63):
            # We'll need to go through lower precision floats
            A = arr.astype(BFT)
            Ai = A - inter
            ulps = [big_bad_ulp(A), big_bad_ulp(Ai)]
            exp_abs_err = np.max(ulps, axis=0)
        else: # we don't have to go through floats - no error !
            exp_abs_err = np.zeros_like(abs_err)
        rel_thresh = 0
    else:
        # Error from integer rounding
        inting_err = np.abs(scaling_type(slope) / 2)
        inting_err = inting_err + ulp(inting_err)
        # Error from calculation of inter
        inter_err = ulp(scaling_type(inter))
        # Max abs error from floating point
        Ai = arr - scaling_type(inter)
        Ais = Ai / scaling_type(slope)
        exp_abs_err = inting_err + inter_err + (
            big_bad_ulp(Ai) + big_bad_ulp(Ais))
        # Relative scaling error from calculation of slope
        # This threshold needs to be 2 x larger on windows 32 bit and PPC for
        # some reason
        rel_thresh = ulp(scaling_type(1))
    test_vals = (abs_err <= exp_abs_err) | (rel_err <= rel_thresh)
    this_test = np.all(test_vals)
    if DEBUG:
        abs_fails = (abs_err > exp_abs_err)
        rel_fails = (rel_err > rel_thresh)
        all_fails = abs_fails & rel_fails
        if np.any(rel_fails):
            abs_mx_e = abs_err[rel_fails].max()
            exp_abs_mx_e = exp_abs_err[rel_fails].max()
        else:
            abs_mx_e = None
            exp_abs_mx_e = None
        if np.any(abs_fails):
            rel_mx_e = rel_err[abs_fails].max()
        else:
            rel_mx_e = None
        print (test_id,
               np.dtype(in_type).str,
               np.dtype(out_type).str,
               exp_abs_mx_e,
               abs_mx_e,
               rel_thresh,
               rel_mx_e,
               slope, inter)
        # To help debugging failures with --pdb-failure
        fail_i = np.nonzero(all_fails)
    assert_true(this_test)
