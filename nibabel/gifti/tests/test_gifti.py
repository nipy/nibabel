""" Testing gifti objects
"""

import numpy as np

from ...nifti1 import data_type_codes, intent_codes

from ..gifti import GiftiImage, GiftiDataArray

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_equal, assert_raises


def test_gifti_image():
    # Check that we're not modifying the default empty list in the default
    # arguments.
    gi = GiftiImage()
    assert_equal(gi.darrays, [])
    arr = np.zeros((2,3))
    gi.darrays.append(arr)
    # Now check we didn't overwrite the default arg
    gi = GiftiImage()
    assert_equal(gi.darrays, [])


def test_dataarray():
    for dt_code in data_type_codes.value_set():
        data_type = data_type_codes.type[dt_code]
        if data_type is np.void: # not supported
            continue
        arr = np.zeros((10,3), dtype=data_type)
        da = GiftiDataArray.from_array(arr, 'triangle')
        assert_equal(da.datatype, data_type_codes[arr.dtype])
        bs_arr = arr.byteswap().newbyteorder()
        da = GiftiDataArray.from_array(bs_arr, 'triangle')
        assert_equal(da.datatype, data_type_codes[arr.dtype])
