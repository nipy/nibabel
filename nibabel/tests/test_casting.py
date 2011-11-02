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
            iarr = nice_round(farr, it)
            mn, mx = int_clippers(ft, it)
            assert_array_equal(iarr, [mn, mx, mn, mx, 0, 0, 11])
