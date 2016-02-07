# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Testing mriutils module
"""
from __future__ import division


from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


from ..mriutils import calculate_dwell_time, MRIError


def test_calculate_dwell_time():
    # Test dwell time calculation
    # This tests only that the calculation does what it appears to; needs some
    # external check
    assert_almost_equal(calculate_dwell_time(3.3, 2, 3),
                        3.3 / (42.576 * 3.4 * 3 * 3))
    # Echo train length of 1 is valid, but returns 0 dwell time
    assert_almost_equal(calculate_dwell_time(3.3, 1, 3), 0)
    assert_raises(MRIError, calculate_dwell_time, 3.3, 0, 3.0)
    assert_raises(MRIError, calculate_dwell_time, 3.3, 2, -0.1)
