""" Test casting utilities
"""

import numpy as np

from ..casting import nice_round, int_clippers

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_equal, assert_raises)


def test_int_clippers():
    for ft in np.sctypes['float']:
        for it in np.sctypes['int'] + np.sctypes['uint']:
            # Test that going a bit above or below the calculated min and max
            # either generates the same number when cast, or something smaller
            # (as a result of overflow)
            mn, mx = int_clippers(ft, it)
            ovs = ft(mx) + np.arange(2048, dtype=ft)
            # Float16 can overflow to inf
            bit_bigger = ovs[np.isfinite(ovs)].astype(it)
            casted_mx = ft(mx).astype(it)
            assert_true(np.all((bit_bigger <= casted_mx)))
            if it in np.sctypes['uint']:
                assert_equal(mn, 0)
                continue
            # And something larger for the minimum
            ovs = ft(mn) - np.arange(2048, dtype=ft)
            # Float16 can overflow to inf
            bit_smaller = ovs[np.isfinite(ovs)].astype(it)
            casted_mn = ft(mn).astype(it)
            assert_true(np.all(bit_smaller >= casted_mn))


def test_casting():
    for ft in np.sctypes['float']:
        for it in np.sctypes['int'] + np.sctypes['uint']:
            ii = np.iinfo(it)
            arr = [ii.min-1, ii.max+1, -np.inf, np.inf, np.nan, 0.2, 10.6]
            farr = np.array(arr, dtype=ft)
            mn, mx = int_clippers(ft, it)
            iarr = nice_round(farr, it)
            exp_arr = np.array([mn, mx, mn, mx, 0, 0, 11])
            assert_array_equal(iarr, exp_arr)
            iarr = nice_round(farr, it, infmax=True)
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
            assert_array_equal(farr, np.array(arr, dtype=ft))
    # Test scalars work and return scalars
    assert_array_equal(nice_round(np.float32(0), np.int16), [0])
