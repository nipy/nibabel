""" Test casting utilities
"""

import numpy as np

from ..casting import nice_round, int_clippers

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_equal, assert_raises)


def test_int_clippers():
    for ft in np.sctypes['float']:
        if ft == np.longdouble:
            # A bit difficult to test elegantly, because assignment and
            # comparison go through float64 or maybe uint64 - but in any case
            # they need fixing for tests like these
            continue
        for it in np.sctypes['int'] + np.sctypes['uint']:
            mn, mx = int_clippers(ft, it)
            ovs = ft(mx) + np.arange(1024, dtype=ft)
            assert_true(np.all(ovs[ovs > mx] > np.iinfo(it).max))
            ovs = ft(mn) - np.arange(1024, dtype=ft)
            assert_true(np.all(ovs[ovs < mn] < np.iinfo(it).min))


def test_casting():
    for ft in np.sctypes['float']:
        for it in np.sctypes['int'] + np.sctypes['uint']:
            ii = np.iinfo(it)
            arr = [ii.min-1, ii.max+1, -np.inf, np.inf, np.nan, 0.2, 10.6]
            farr = np.array(arr, dtype=ft)
            iarr = nice_round(farr, it)
            mn, mx = int_clippers(ft, it)
            assert_array_equal(iarr, [mn, mx, mn, mx, 0, 0, 11])
