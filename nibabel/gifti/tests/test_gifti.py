""" Testing gifti objects
"""

import numpy as np

from ..gifti import GiftiImage

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_equal, assert_raises


def test_gifti_data():
    # Check that we're not modifying the default empty list in the default
    # arguments.
    gi = GiftiImage()
    assert_equal(gi.darrays, [])
    arr = np.zeros((2,3))
    gi.darrays.append(arr)
    # Now check we didn't overwrite the default arg
    gi = GiftiImage()
    assert_equal(gi.darrays, [])

