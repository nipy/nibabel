""" Test casting utilities
"""
import os

from platform import machine

import numpy as np

from ..casting import (float_to_int, shared_range, CastingError, int_to_float,
                       as_int, int_abs, floor_log2, able_int_type, best_float,
                       ulp, longdouble_precision_improved)

from numpy.testing import (assert_array_almost_equal, assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)


def test_shared_range():
    for ft in np.sctypes['float']:
        for it in np.sctypes['int'] + np.sctypes['uint']:
            # Test that going a bit above or below the calculated min and max
            # either generates the same number when cast, or the max int value
            # (if this system generates that) or something smaller (because of
            # overflow)
            mn, mx = shared_range(ft, it)
            ovs = ft(mx) + np.arange(2048, dtype=ft)
            # Float16 can overflow to inf
            bit_bigger = ovs[np.isfinite(ovs)].astype(it)
            casted_mx = ft(mx).astype(it)
            imax = int(np.iinfo(it).max)
            thresh_overflow = False
            if casted_mx != imax:
                # The shared_range have told us that they believe the imax does
                # not have an exact representation.
                fimax = int_to_float(imax, ft)
                if np.isfinite(fimax):
                    assert_true(int(fimax) != imax)
                # Therefore the imax, cast back to float, and to integer, will
                # overflow. If it overflows to the imax, we need to allow for
                # that possibility in the testing of our overflowed values
                imax_roundtrip = fimax.astype(it)
                if imax_roundtrip == imax:
                    thresh_overflow = True
            if thresh_overflow:
                assert_true(np.all(
                    (bit_bigger == casted_mx) |
                    (bit_bigger == imax)))
            else:
                assert_true(np.all((bit_bigger <= casted_mx)))
            if it in np.sctypes['uint']:
                assert_equal(mn, 0)
                continue
            # And something larger for the minimum
            ovs = ft(mn) - np.arange(2048, dtype=ft)
            # Float16 can overflow to inf
            bit_smaller = ovs[np.isfinite(ovs)].astype(it)
            casted_mn = ft(mn).astype(it)
            imin = int(np.iinfo(it).min)
            if casted_mn != imin:
                # The shared_range have told us that they believe the imin does
                # not have an exact representation.
                fimin = int_to_float(imin, ft)
                if np.isfinite(fimin):
                    assert_true(int(fimin) != imin)
                # Therefore the imin, cast back to float, and to integer, will
                # overflow. If it overflows to the imin, we need to allow for
                # that possibility in the testing of our overflowed values
                imin_roundtrip = fimin.astype(it)
                if imin_roundtrip == imin:
                    thresh_overflow = True
            if thresh_overflow:
                assert_true(np.all(
                    (bit_smaller == casted_mn) |
                    (bit_smaller == imin)))
            else:
                assert_true(np.all((bit_smaller >= casted_mn)))


def test_shared_range_inputs():
    # Check any dtype specifier will work as input
    rng0 = shared_range(np.float32, np.int32)
    assert_array_equal(rng0, shared_range('f4', 'i4'))
    assert_array_equal(rng0, shared_range(np.dtype('f4'), np.dtype('i4')))


def test_casting():
    for ft in np.sctypes['float']:
        for it in np.sctypes['int'] + np.sctypes['uint']:
            ii = np.iinfo(it)
            arr = [ii.min-1, ii.max+1, -np.inf, np.inf, np.nan, 0.2, 10.6]
            farr_orig = np.array(arr, dtype=ft)
            # We're later going to test if we modify this array
            farr = farr_orig.copy()
            mn, mx = shared_range(ft, it)
            iarr = float_to_int(farr, it)
            # Dammit - for long doubles we need to jump through some hoops not
            # to round to numbers outside the range
            if ft is np.longdouble:
                mn = as_int(mn)
                mx = as_int(mx)
            exp_arr = np.array([mn, mx, mn, mx, 0, 0, 11], dtype=it)
            assert_array_equal(iarr, exp_arr)
            # Now test infmax version
            iarr = float_to_int(farr, it, infmax=True)
            im_exp = np.array([mn, mx, ii.min, ii.max, 0, 0, 11], dtype=it)
            # Float16 can overflow to infs
            if farr[0] == -np.inf:
                im_exp[0] = ii.min
            if farr[1] == np.inf:
                im_exp[1] = ii.max
            assert_array_equal(iarr, im_exp)
            # NaNs, with nan2zero False, gives error
            assert_raises(CastingError, float_to_int, farr, it, False)
            # We can pass through NaNs if we really want
            exp_arr[arr.index(np.nan)] = ft(np.nan).astype(it)
            iarr = float_to_int(farr, it, nan2zero=None)
            assert_array_equal(iarr, exp_arr)
            # Confirm input array is not modified
            nans = np.isnan(farr)
            assert_array_equal(nans, np.isnan(farr_orig))
            assert_array_equal(farr[nans==False], farr_orig[nans==False])
    # Test scalars work and return scalars
    assert_array_equal(float_to_int(np.float32(0), np.int16), [0])
    # Test scalar nan OK
    assert_array_equal(float_to_int(np.nan, np.int16), [0])
    # Test nans give error if not nan2zero
    assert_raises(CastingError, float_to_int, np.nan, np.int16, False)


def test_int_abs():
    for itype in np.sctypes['int']:
        info = np.iinfo(itype)
        in_arr = np.array([info.min, info.max], dtype=itype)
        idtype = np.dtype(itype)
        udtype = np.dtype(idtype.str.replace('i', 'u'))
        assert_equal(udtype.kind, 'u')
        assert_equal(idtype.itemsize, udtype.itemsize)
        mn, mx = in_arr
        e_mn = as_int(mx) + 1 # as_int needed for numpy 1.4.1 casting
        assert_equal(int_abs(mx), mx)
        assert_equal(int_abs(mn), e_mn)
        assert_array_equal(int_abs(in_arr), [e_mn, mx])


def test_floor_log2():
    assert_equal(floor_log2(2**9+1), 9)
    assert_equal(floor_log2(-2**9+1), 8)
    assert_equal(floor_log2(2), 1)
    assert_equal(floor_log2(1), 0)
    assert_equal(floor_log2(0.5), -1)
    assert_equal(floor_log2(0.75), -1)
    assert_equal(floor_log2(0.25), -2)
    assert_equal(floor_log2(0.24), -3)
    assert_equal(floor_log2(0), None)


def test_able_int_type():
    # The integer type cabable of containing values
    for vals, exp_out in (
        ([0, 1], np.uint8),
        ([0, 255], np.uint8),
        ([-1, 1], np.int8),
        ([0, 256], np.uint16),
        ([-1, 128], np.int16),
        ([0.1, 1], None),
        ([0, 2**16], np.uint32),
        ([-1, 2**15], np.int32),
        ([0, 2**32], np.uint64),
        ([-1, 2**31], np.int64),
        ([-1, 2**64-1], None),
        ([0, 2**64-1], np.uint64),
        ([0, 2**64], None)):
        assert_equal(able_int_type(vals), exp_out)


def test_able_casting():
    # Check the able_int_type function guesses numpy out type
    types = np.sctypes['int'] + np.sctypes['uint']
    for in_type in types:
        in_info = np.iinfo(in_type)
        in_mn, in_mx = in_info.min, in_info.max
        A = np.zeros((1,), dtype=in_type)
        for out_type in types:
            out_info = np.iinfo(out_type)
            out_mn, out_mx = out_info.min, out_info.max
            B = np.zeros((1,), dtype=out_type)
            ApBt = (A + B).dtype.type
            able_type = able_int_type([in_mn, in_mx, out_mn, out_mx])
            if able_type is None:
                assert_equal(ApBt, np.float64)
                continue
            # Use str for comparison to avoid int32/64 vs intp comparison
            # failures
            assert_equal(np.dtype(ApBt).str, np.dtype(able_type).str)


def test_best_float():
    # Finds the most capable floating point type
    """ most capable type will be np.longdouble except when

    * np.longdouble has float64 precision (MSVC compiled numpy)
    * machine is sparc64 (float128 very slow)
    * np.longdouble had float64 precision when ``casting`` moduled was imported
     (precisions on windows can change, apparently)
    """
    best = best_float()
    end_of_ints = np.float64(2**53)
    # float64 has continuous integers up to 2**53
    assert_equal(end_of_ints, end_of_ints + 1)
    # longdouble may have more, but not on 32 bit windows, at least
    end_of_ints = np.longdouble(2**53)
    if (end_of_ints == (end_of_ints + 1) or # off continuous integers
        machine() == 'sparc64' or # crippling slow longdouble on sparc
        longdouble_precision_improved()): # Windows precisions can change
        assert_equal(best, np.float64)
    else:
        assert_equal(best, np.longdouble)


def test_longdouble_precision_improved():
    # Just check that this can only be True on windows, msvc
    from numpy.distutils.ccompiler import get_default_compiler
    if not (os.name == 'nt' and get_default_compiler() == 'msvc'):
        assert_false(longdouble_precision_improved())


def test_ulp():
    assert_equal(ulp(), np.finfo(np.float64).eps)
    assert_equal(ulp(1.0), np.finfo(np.float64).eps)
    assert_equal(ulp(np.float32(1.0)), np.finfo(np.float32).eps)
    assert_equal(ulp(np.float32(1.999)), np.finfo(np.float32).eps)
    # Integers always return 1
    assert_equal(ulp(1), 1)
    assert_equal(ulp(2**63-1), 1)
    # negative / positive same
    assert_equal(ulp(-1), 1)
    assert_equal(ulp(7.999), ulp(4.0))
    assert_equal(ulp(-7.999), ulp(4.0))
    assert_equal(ulp(np.float64(2**54-2)), 2)
    assert_equal(ulp(np.float64(2**54)), 4)
    assert_equal(ulp(np.float64(2**54)), 4)
    # Infs, NaNs return nan
    assert_true(np.isnan(ulp(np.inf)))
    assert_true(np.isnan(ulp(-np.inf)))
    assert_true(np.isnan(ulp(np.nan)))
    # 0 gives subnormal smallest
    subn64 = np.float64(2**(-1022-52))
    subn32 = np.float32(2**(-126-23))
    assert_equal(ulp(0.0), subn64)
    assert_equal(ulp(np.float64(0)), subn64)
    assert_equal(ulp(np.float32(0)), subn32)
    # as do multiples of subnormal smallest
    assert_equal(ulp(subn64 * np.float64(2**52)), subn64)
    assert_equal(ulp(subn64 * np.float64(2**53)), subn64*2)
    assert_equal(ulp(subn32 * np.float32(2**23)), subn32)
    assert_equal(ulp(subn32 * np.float32(2**24)), subn32*2)
