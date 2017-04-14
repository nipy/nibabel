""" Test for utils module
"""

import numpy as np

from nibabel.utils import to_scalar

from nose.tools import assert_equal, assert_true, assert_false, assert_raises


def test_to_scalar():
    for pass_thru in (2, 2.3, 'foo', b'foo', [], (), [2], (2,), object()):
        assert_true(to_scalar(pass_thru) is pass_thru)
    for arr_contents in (2, 2.3, 'foo', b'foo'):
        arr = np.array(arr_contents)
        out = to_scalar(arr)
        assert_false(to_scalar(arr) is arr)
        assert_equal(out, arr_contents)
        # Promote to 1 and 2D and check contents
        assert_equal(to_scalar(np.atleast_1d(arr)), arr_contents)
        assert_equal(to_scalar(np.atleast_2d(arr)), arr_contents)
    assert_raises(ValueError, to_scalar, np.array([1, 2]))
