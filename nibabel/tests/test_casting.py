""" Test casting utilities
"""

import numpy as np

from ..casting import (float_to_int, int_clippers, CastingError, int_to_float)

from numpy.testing import (assert_array_almost_equal, assert_array_equal)

from nose.tools import (assert_true, assert_equal, assert_raises)


def test_int_clippers():
    for ft in np.sctypes['float']:
        for it in np.sctypes['int'] + np.sctypes['uint']:
            # Test that going a bit above or below the calculated min and max
            # either generates the same number when cast, or the max int value
            # (if this system generates that) or something smaller (because of
            # overflow)
            mn, mx = int_clippers(ft, it)
            ovs = ft(mx) + np.arange(2048, dtype=ft)
            # Float16 can overflow to inf
            bit_bigger = ovs[np.isfinite(ovs)].astype(it)
            casted_mx = ft(mx).astype(it)
            imax = int(np.iinfo(it).max)
            thresh_overflow = False
            if casted_mx != imax:
                # The int_clippers have told us that they believe the imax does
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
                # The int_clippers have told us that they believe the imin does
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


def test_casting():
    for ft in np.sctypes['float']:
        for it in np.sctypes['int'] + np.sctypes['uint']:
            ii = np.iinfo(it)
            arr = [ii.min-1, ii.max+1, -np.inf, np.inf, np.nan, 0.2, 10.6]
            farr_orig = np.array(arr, dtype=ft)
            # We're later going to test if we modify this array
            farr = farr_orig.copy()
            mn, mx = int_clippers(ft, it)
            iarr = float_to_int(farr, it)
            exp_arr = np.array([mn, mx, mn, mx, 0, 0, 11])
            assert_array_equal(iarr, exp_arr)
            iarr = float_to_int(farr, it, infmax=True)
            # Float16 can overflow to infs
            if farr[0] == -np.inf:
                exp_arr[0] = ii.min
            if farr[1] == np.inf:
                exp_arr[1] = ii.max
            exp_arr[2] = ii.min
            if exp_arr.dtype.type is np.longdouble:
                # longdouble seems to go through float64 on assignment; if
                # ii.max is above float64 integer resolution we have go through
                # float64 to split up the number and get full precision
                f64 = np.float64(ii.max)
                exp_arr[3] = np.longdouble(f64) + np.float64(ii.max - int(f64))
            else:
                exp_arr[3] = ii.max
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
