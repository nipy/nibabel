""" Testing spatialimages

"""

import numpy as np

from nibabel.spatialimages import Header, SpatialImage, \
    HeaderDataError, ImageDataError, ImageFileError

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nibabel.testing import data_path, parametric


@parametric
def test_header_init():
    # test the basic header
    hdr = Header()
    yield assert_equal(hdr.get_io_dtype(), np.dtype(np.float32))
    yield assert_equal(hdr.get_data_shape(), (0,))
    yield assert_equal(hdr.get_zooms(), (1.0,))
    hdr = Header(np.float64)
    yield assert_equal(hdr.get_io_dtype(), np.dtype(np.float64))
    yield assert_equal(hdr.get_data_shape(), (0,))
    yield assert_equal(hdr.get_zooms(), (1.0,))
    hdr = Header(np.float64, shape=(1,2,3))
    yield assert_equal(hdr.get_io_dtype(), np.dtype(np.float64))
    yield assert_equal(hdr.get_data_shape(), (1,2,3))
    yield assert_equal(hdr.get_zooms(), (1.0, 1.0, 1.0))
    hdr = Header(np.float64, shape=(1,2,3), zooms=None)
    yield assert_equal(hdr.get_io_dtype(), np.dtype(np.float64))
    yield assert_equal(hdr.get_data_shape(), (1,2,3))
    yield assert_equal(hdr.get_zooms(), (1.0, 1.0, 1.0))
    hdr = Header(np.float64, shape=(1,2,3), zooms=(3.0, 2.0, 1.0))
    yield assert_equal(hdr.get_io_dtype(), np.dtype(np.float64))
    yield assert_equal(hdr.get_data_shape(), (1,2,3))
    yield assert_equal(hdr.get_zooms(), (3.0, 2.0, 1.0))


@parametric
def test_shape_zooms():
    hdr = Header()
    hdr.set_data_shape((1, 2, 3))
    yield assert_equal(hdr.get_data_shape(), (1,2,3))
    yield assert_equal(hdr.get_zooms(), (1.0,1.0,1.0))
    hdr.set_zooms((4, 3, 2))
    yield assert_equal(hdr.get_zooms(), (4.0,3.0,2.0))    
    hdr.set_data_shape((1, 2))
    yield assert_equal(hdr.get_data_shape(), (1,2))
    yield assert_equal(hdr.get_zooms(), (4.0,3.0))
    hdr.set_data_shape((1, 2, 3))
    yield assert_equal(hdr.get_data_shape(), (1,2,3))
    yield assert_equal(hdr.get_zooms(), (4.0,3.0,1.0))
    # null shape is (0,)
    hdr.set_data_shape(())
    yield assert_equal(hdr.get_data_shape(), (0,))
    yield assert_equal(hdr.get_zooms(), (1.0,))
    # zooms of wrong lengths raise error
    yield assert_raises(HeaderDataError, hdr.set_zooms, (4.0, 3.0))
    yield assert_raises(HeaderDataError, hdr.set_zooms, (4.0, 3.0, 2.0, 1.0))


@parametric
def test_io_dtype():
    hdr = Header()
    yield assert_equal(hdr.get_io_dtype(), np.dtype(np.float32))
    hdr.set_io_dtype(np.float64)
    yield assert_equal(hdr.get_io_dtype(), np.dtype(np.float64))
    hdr.set_io_dtype('u2')
    yield assert_equal(hdr.get_io_dtype(), np.dtype(np.uint16))


@parametric
def test_affine():
    pass
