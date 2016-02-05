# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Test for scaling / rounding in volumeutils module '''
from __future__ import division, print_function, absolute_import

import numpy as np

from ..externals.six import BytesIO
from ..volumeutils import (calculate_scale, scale_min_max, finite_range,
                           apply_read_scaling, array_to_file, array_from_file)
from ..casting import type_info
from ..testing import suppress_warnings

from numpy.testing import (assert_array_almost_equal, assert_array_equal)

from nose.tools import (assert_true, assert_equal, assert_raises,
                        assert_not_equal)


# Debug print statements
DEBUG = True


def test_scale_min_max():
    mx_dt = np.maximum_sctype(np.float)
    for tp in np.sctypes['uint'] + np.sctypes['int']:
        info = np.iinfo(tp)
        # Need to pump up to max fp type to contain python longs
        imin = np.array(info.min, dtype=mx_dt)
        imax = np.array(info.max, dtype=mx_dt)
        value_pairs = (
            (0, imax),
            (imin, 0),
            (imin, imax),
            (1, 10),
            (-1, -1),
            (1, 1),
            (-10, -1),
            (-100, 10))
        for mn, mx in value_pairs:
            # with intercept
            scale, inter = scale_min_max(mn, mx, tp, True)
            if mx - mn:
                assert_array_almost_equal, (mx - inter) / scale, imax
                assert_array_almost_equal, (mn - inter) / scale, imin
            else:
                assert_equal, (scale, inter), (1.0, mn)
            # without intercept
            if imin == 0 and mn < 0 and mx > 0:
                (assert_raises, ValueError,
                 scale_min_max, mn, mx, tp, False)
                continue
            scale, inter = scale_min_max(mn, mx, tp, False)
            assert_equal, inter, 0.0
            if mn == 0 and mx == 0:
                assert_equal, scale, 1.0
                continue
            sc_mn = mn / scale
            sc_mx = mx / scale
            assert_true, sc_mn >= imin
            assert_true, sc_mx <= imax
            if imin == 0:
                if mx > 0:  # numbers all +ve
                    assert_array_almost_equal, mx / scale, imax
                else:  # numbers all -ve
                    assert_array_almost_equal, mn / scale, imax
                continue
            if abs(mx) >= abs(mn):
                assert_array_almost_equal, mx / scale, imax
            else:
                assert_array_almost_equal, mn / scale, imin


def test_finite_range():
    # Finite range utility function
    for in_arr, res in (
        ([[-1, 0, 1], [np.inf, np.nan, -np.inf]], (-1, 1)),
        (np.array([[-1, 0, 1], [np.inf, np.nan, -np.inf]]), (-1, 1)),
        ([[np.nan], [np.nan]], (np.inf, -np.inf)),  # all nans slices
        (np.zeros((3, 4, 5)) + np.nan, (np.inf, -np.inf)),
        ([[-np.inf], [np.inf]], (np.inf, -np.inf)),  # all infs slices
        (np.zeros((3, 4, 5)) + np.inf, (np.inf, -np.inf)),
        ([[np.nan, -1, 2], [-2, np.nan, 1]], (-2, 2)),
        ([[np.nan, -np.inf, 2], [-2, np.nan, np.inf]], (-2, 2)),
        ([[-np.inf, 2], [np.nan, 1]], (1, 2)),  # good max case
        ([[np.nan, -np.inf, 2], [-2, np.nan, np.inf]], (-2, 2)),
        ([np.nan], (np.inf, -np.inf)),
        ([np.inf], (np.inf, -np.inf)),
        ([-np.inf], (np.inf, -np.inf)),
        ([np.inf, 1], (1, 1)),  # only look at finite values
        ([-np.inf, 1], (1, 1)),
        ([[], []], (np.inf, -np.inf)),  # empty array
        (np.array([[-3, 0, 1], [2, -1, 4]], dtype=np.int), (-3, 4)),
        (np.array([[1, 0, 1], [2, 3, 4]], dtype=np.uint), (0, 4)),
        ([0., 1, 2, 3], (0, 3)),
        # Complex comparison works as if they are floats
        ([[np.nan, -1 - 100j, 2], [-2, np.nan, 1 + 100j]], (-2, 2)),
        ([[np.nan, -1, 2 - 100j], [-2 + 100j, np.nan, 1]], (-2 + 100j, 2 - 100j)),
    ):
        assert_equal(finite_range(in_arr), res)
        assert_equal(finite_range(in_arr, False), res)
        assert_equal(finite_range(in_arr, check_nan=False), res)
        has_nan = np.any(np.isnan(in_arr))
        assert_equal(finite_range(in_arr, True), res + (has_nan,))
        assert_equal(finite_range(in_arr, check_nan=True), res + (has_nan,))
        in_arr = np.array(in_arr)
        flat_arr = in_arr.ravel()
        assert_equal(finite_range(flat_arr), res)
        assert_equal(finite_range(flat_arr, True), res + (has_nan,))
        # Check float types work as complex
        if in_arr.dtype.kind == 'f':
            c_arr = in_arr.astype(np.complex)
            assert_equal(finite_range(c_arr), res)
            assert_equal(finite_range(c_arr, True), res + (has_nan,))
    # Test error cases
    a = np.array([[1., 0, 1], [2, 3, 4]]).view([('f1', 'f')])
    assert_raises(TypeError, finite_range, a)


def test_calculate_scale():
    # Test for special cases in scale calculation
    npa = np.array
    # Here the offset handles it
    res = calculate_scale(npa([-2, -1], dtype=np.int8), np.uint8, True)
    assert_equal(res, (1.0, -2.0, None, None))
    # Not having offset not a problem obviously
    res = calculate_scale(npa([-2, -1], dtype=np.int8), np.uint8, 0)
    assert_equal(res, (-1.0, 0.0, None, None))
    # Case where offset handles scaling
    res = calculate_scale(npa([-1, 1], dtype=np.int8), np.uint8, 1)
    assert_equal(res, (1.0, -1.0, None, None))
    # Can't work for no offset case
    assert_raises(ValueError,
                  calculate_scale, npa([-1, 1], dtype=np.int8), np.uint8, 0)
    # Offset trick can't work when max is out of range
    res = calculate_scale(npa([-1, 255], dtype=np.int16), np.uint8, 1)
    assert_not_equal(res, (1.0, -1.0, None, None))


def test_a2f_mn_mx():
    # Test array to file mn, mx handling
    str_io = BytesIO()
    for out_type in (np.int16, np.float32):
        arr = np.arange(6, dtype=out_type)
        arr_orig = arr.copy()  # safe backup for testing against
        # Basic round trip to warm up
        array_to_file(arr, str_io)
        data_back = array_from_file(arr.shape, out_type, str_io)
        assert_array_equal(arr, data_back)
        # Clip low
        array_to_file(arr, str_io, mn=2)
        data_back = array_from_file(arr.shape, out_type, str_io)
        # arr unchanged
        assert_array_equal(arr, arr_orig)
        # returned value clipped low
        assert_array_equal(data_back, [2, 2, 2, 3, 4, 5])
        # Clip high
        array_to_file(arr, str_io, mx=4)
        data_back = array_from_file(arr.shape, out_type, str_io)
        # arr unchanged
        assert_array_equal(arr, arr_orig)
        # returned value clipped high
        assert_array_equal(data_back, [0, 1, 2, 3, 4, 4])
        # Clip both
        array_to_file(arr, str_io, mn=2, mx=4)
        data_back = array_from_file(arr.shape, out_type, str_io)
        # arr unchanged
        assert_array_equal(arr, arr_orig)
        # returned value clipped high
        assert_array_equal(data_back, [2, 2, 2, 3, 4, 4])


def test_a2f_nan2zero():
    # Test conditions under which nans written to zero
    arr = np.array([np.nan, 99.], dtype=np.float32)
    str_io = BytesIO()
    array_to_file(arr, str_io)
    data_back = array_from_file(arr.shape, np.float32, str_io)
    assert_array_equal(np.isnan(data_back), [True, False])
    # nan2zero ignored for floats
    array_to_file(arr, str_io, nan2zero=True)
    data_back = array_from_file(arr.shape, np.float32, str_io)
    assert_array_equal(np.isnan(data_back), [True, False])
    # Integer output with nan2zero gives zero
    with np.errstate(invalid='ignore'):
        array_to_file(arr, str_io, np.int32, nan2zero=True)
    data_back = array_from_file(arr.shape, np.int32, str_io)
    assert_array_equal(data_back, [0, 99])
    # Integer output with nan2zero=False gives whatever astype gives
    with np.errstate(invalid='ignore'):
        array_to_file(arr, str_io, np.int32, nan2zero=False)
    data_back = array_from_file(arr.shape, np.int32, str_io)
    assert_array_equal(data_back, [np.array(np.nan).astype(np.int32), 99])


def test_array_file_scales():
    # Test scaling works for max, min when going from larger to smaller type,
    # and from float to integer.
    bio = BytesIO()
    for in_type, out_type, err in ((np.int16, np.int16, None),
                                   (np.int16, np.int8, None),
                                   (np.uint16, np.uint8, None),
                                   (np.int32, np.int8, None),
                                   (np.float32, np.uint8, None),
                                   (np.float32, np.int16, None)):
        out_dtype = np.dtype(out_type)
        arr = np.zeros((3,), dtype=in_type)
        info = type_info(in_type)
        arr[0], arr[1] = info['min'], info['max']
        if not err is None:
            assert_raises(err, calculate_scale, arr, out_dtype, True)
            continue
        slope, inter, mn, mx = calculate_scale(arr, out_dtype, True)
        array_to_file(arr, bio, out_type, 0, inter, slope, mn, mx)
        bio.seek(0)
        arr2 = array_from_file(arr.shape, out_dtype, bio)
        arr3 = apply_read_scaling(arr2, slope, inter)
        # Max rounding error for integer type
        max_miss = slope / 2.
        assert_true(np.all(np.abs(arr - arr3) <= max_miss))
        bio.truncate(0)
        bio.seek(0)


def test_scaling_in_abstract():
    # Confirm that, for all ints and uints as input, and all possible outputs,
    # for any simple way of doing the calculation, the result is near enough
    for category0, category1 in (('int', 'int'),
                                 ('uint', 'int'),
                                 ):
        for in_type in np.sctypes[category0]:
            for out_type in np.sctypes[category1]:
                check_int_a2f(in_type, out_type)
    # Converting floats to integer
    for category0, category1 in (('float', 'int'),
                                 ('float', 'uint'),
                                 ('complex', 'int'),
                                 ('complex', 'uint'),
                                 ):
        for in_type in np.sctypes[category0]:
            for out_type in np.sctypes[category1]:
                with suppress_warnings():  # overflow
                    check_int_a2f(in_type, out_type)


def check_int_a2f(in_type, out_type):
    # Check that array to / from file returns roughly the same as input
    big_floater = np.maximum_sctype(np.float)
    info = type_info(in_type)
    this_min, this_max = info['min'], info['max']
    if not in_type in np.sctypes['complex']:
        data = np.array([this_min, this_max], in_type)
        # Bug in numpy 1.6.2 on PPC leading to infs - abort
        if not np.all(np.isfinite(data)):
            if DEBUG:
                print('Hit PPC max -> inf bug; skip in_type %s' % in_type)
            return
    else:  # Funny behavior with complex256
        data = np.zeros((2,), in_type)
        data[0] = this_min + 0j
        data[1] = this_max + 0j
    str_io = BytesIO()
    try:
        scale, inter, mn, mx = calculate_scale(data, out_type, True)
    except ValueError as e:
        if DEBUG:
            print(in_type, out_type, e)
        return
    array_to_file(data, str_io, out_type, 0, inter, scale, mn, mx)
    data_back = array_from_file(data.shape, out_type, str_io)
    data_back = apply_read_scaling(data_back, scale, inter)
    assert_true(np.allclose(big_floater(data), big_floater(data_back)))
    # Try with analyze-size scale and inter
    scale32 = np.float32(scale)
    inter32 = np.float32(inter)
    if scale32 == np.inf or inter32 == np.inf:
        return
    data_back = array_from_file(data.shape, out_type, str_io)
    data_back = apply_read_scaling(data_back, scale32, inter32)
    # Clip at extremes to remove inf
    info = type_info(in_type)
    out_min, out_max = info['min'], info['max']
    assert_true(np.allclose(big_floater(data),
                            big_floater(np.clip(data_back, out_min, out_max))))
