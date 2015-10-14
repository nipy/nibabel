""" Testing gifti objects
"""
import warnings

import numpy as np

from ..gifti import (GiftiImage, GiftiDataArray, GiftiLabel, GiftiLabelTable,
                     GiftiMetaData)
from ...nifti1 import data_type_codes, intent_codes

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)
from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)
from ...testing import clear_and_catch_warnings


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

    # Test darrays / numDA
    gi = GiftiImage()
    assert_equal(gi.numDA, 0)

    da = GiftiDataArray(data='data')
    gi.add_gifti_data_array(da)
    assert_equal(gi.numDA, 1)
    assert_equal(gi.darrays[0].data, 'data')

    gi.remove_gifti_data_array(0)
    assert_equal(gi.numDA, 0)

    # Remove from empty
    gi = GiftiImage()
    gi.remove_gifti_data_array_by_intent(0)
    assert_equal(gi.numDA, 0)

    # Remove one
    gi = GiftiImage()
    da = GiftiDataArray(data='data')
    gi.add_gifti_data_array(da)

    gi.remove_gifti_data_array_by_intent(0)
    assert_equal(gi.numDA, 1)

    gi.darrays[0].intent = 0
    gi.remove_gifti_data_array_by_intent(0)
    assert_equal(gi.numDA, 0)


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


def test_labeltable():
    img = GiftiImage()
    assert_equal(len(img.labeltable.labels), 0)

    new_table = GiftiLabelTable()
    new_table.labels += ['test', 'me']
    img.labeltable = new_table
    assert_equal(len(img.labeltable.labels), 2)

    # Try to set to non-table
    def assign_labeltable(val):
        img.labeltable = val
    assert_raises(ValueError, assign_labeltable, 'not-a-table')


def test_metadata():
    # Test deprecation
    with clear_and_catch_warnings() as w:
        warnings.filterwarnings('once', category=DeprecationWarning)
        assert_equal(len(GiftiDataArray().get_metadata()), 0)

    # Test deprecation
    with clear_and_catch_warnings() as w:
        warnings.filterwarnings('once', category=DeprecationWarning)
        assert_equal(len(GiftiMetaData().get_metadata()), 0)


def test_gifti_label_rgba():
    rgba = np.random.rand(4)
    kwargs = dict(zip(['red', 'green', 'blue', 'alpha'], rgba))

    gl1 = GiftiLabel(**kwargs)
    assert_array_equal(rgba, gl1.rgba)

    gl1.red = 2 * gl1.red
    assert_false(np.allclose(rgba, gl1.rgba))  # don't just store the list!

    gl2 = GiftiLabel()
    gl2.rgba = rgba
    assert_array_equal(rgba, gl2.rgba)

    gl2.blue = 2 * gl2.blue
    assert_false(np.allclose(rgba, gl2.rgba))  # don't just store the list!

    def assign_rgba(gl, val):
        gl.rgba = val
    gl3 = GiftiLabel(**kwargs)
    assert_raises(ValueError, assign_rgba, gl3, rgba[:2])
    assert_raises(ValueError, assign_rgba, gl3, rgba.tolist() + rgba.tolist())

    # Test deprecation
    with clear_and_catch_warnings() as w:
        warnings.filterwarnings('once', category=DeprecationWarning)
        assert_equal(kwargs['red'], gl3.get_rgba()[0])

    # Test default value
    gl4 = GiftiLabel()
    assert_equal(len(gl4.rgba), 4)
    assert_true(np.all([elem is None for elem in gl4.rgba]))
