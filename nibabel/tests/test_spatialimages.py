# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Testing spatialimages

"""
from ..py3k import BytesIO

import numpy as np

from ..spatialimages import (Header, SpatialImage, HeaderDataError,
                             ImageDataError)

from unittest import TestCase

from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_not_equal, assert_raises)

from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_header_init():
    # test the basic header
    hdr = Header()
    assert_equal(hdr.get_data_dtype(), np.dtype(np.float32))
    assert_equal(hdr.get_data_shape(), (0,))
    assert_equal(hdr.get_zooms(), (1.0,))
    hdr = Header(np.float64)
    assert_equal(hdr.get_data_dtype(), np.dtype(np.float64))
    assert_equal(hdr.get_data_shape(), (0,))
    assert_equal(hdr.get_zooms(), (1.0,))
    hdr = Header(np.float64, shape=(1,2,3))
    assert_equal(hdr.get_data_dtype(), np.dtype(np.float64))
    assert_equal(hdr.get_data_shape(), (1,2,3))
    assert_equal(hdr.get_zooms(), (1.0, 1.0, 1.0))
    hdr = Header(np.float64, shape=(1,2,3), zooms=None)
    assert_equal(hdr.get_data_dtype(), np.dtype(np.float64))
    assert_equal(hdr.get_data_shape(), (1,2,3))
    assert_equal(hdr.get_zooms(), (1.0, 1.0, 1.0))
    hdr = Header(np.float64, shape=(1,2,3), zooms=(3.0, 2.0, 1.0))
    assert_equal(hdr.get_data_dtype(), np.dtype(np.float64))
    assert_equal(hdr.get_data_shape(), (1,2,3))
    assert_equal(hdr.get_zooms(), (3.0, 2.0, 1.0))


def test_from_header():
    # check from header class method.  Note equality checks below,
    # equality methods used here too.
    empty = Header.from_header()
    assert_equal(Header(), empty)
    empty = Header.from_header(None)
    assert_equal(Header(), empty)
    hdr = Header(np.float64, shape=(1,2,3), zooms=(3.0, 2.0, 1.0))
    copy = Header.from_header(hdr)
    assert_equal(hdr, copy)
    assert_false(hdr is copy)
    class C(object):
        def get_data_dtype(self): return np.dtype('u2')
        def get_data_shape(self): return (5,4,3)
        def get_zooms(self): return (10.0, 9.0, 8.0)
    converted = Header.from_header(C())
    assert_true(isinstance(converted, Header))
    assert_equal(converted.get_data_dtype(), np.dtype('u2'))
    assert_equal(converted.get_data_shape(), (5,4,3))
    assert_equal(converted.get_zooms(), (10.0,9.0,8.0))


def test_eq():
    hdr = Header()
    other = Header()
    assert_equal(hdr, other)
    other = Header('u2')
    assert_not_equal(hdr, other)
    other = Header(shape=(1,2,3))
    assert_not_equal(hdr, other)
    hdr = Header(shape=(1,2))
    other = Header(shape=(1,2))
    assert_equal(hdr, other)
    other = Header(shape=(1,2), zooms=(2.0,3.0))
    assert_not_equal(hdr, other)


def test_copy():
    # test that copy makes independent copy
    hdr = Header(np.float64, shape=(1,2,3), zooms=(3.0, 2.0, 1.0))
    hdr_copy = hdr.copy()
    hdr.set_data_shape((4,5,6))
    assert_equal(hdr.get_data_shape(), (4,5,6))
    assert_equal(hdr_copy.get_data_shape(), (1,2,3))
    hdr.set_zooms((4,5,6))
    assert_equal(hdr.get_zooms(), (4,5,6))
    assert_equal(hdr_copy.get_zooms(), (3,2,1))
    hdr.set_data_dtype(np.uint8)
    assert_equal(hdr.get_data_dtype(), np.dtype(np.uint8))
    assert_equal(hdr_copy.get_data_dtype(), np.dtype(np.float64))


def test_shape_zooms():
    hdr = Header()
    hdr.set_data_shape((1, 2, 3))
    assert_equal(hdr.get_data_shape(), (1,2,3))
    assert_equal(hdr.get_zooms(), (1.0,1.0,1.0))
    hdr.set_zooms((4, 3, 2))
    assert_equal(hdr.get_zooms(), (4.0,3.0,2.0))    
    hdr.set_data_shape((1, 2))
    assert_equal(hdr.get_data_shape(), (1,2))
    assert_equal(hdr.get_zooms(), (4.0,3.0))
    hdr.set_data_shape((1, 2, 3))
    assert_equal(hdr.get_data_shape(), (1,2,3))
    assert_equal(hdr.get_zooms(), (4.0,3.0,1.0))
    # null shape is (0,)
    hdr.set_data_shape(())
    assert_equal(hdr.get_data_shape(), (0,))
    assert_equal(hdr.get_zooms(), (1.0,))
    # zooms of wrong lengths raise error
    assert_raises(HeaderDataError, hdr.set_zooms, (4.0, 3.0))
    assert_raises(HeaderDataError,
                        hdr.set_zooms,
                        (4.0, 3.0, 2.0, 1.0))
    # as do negative zooms
    assert_raises(HeaderDataError,
                        hdr.set_zooms,
                        (4.0, 3.0, -2.0))


def test_data_dtype():
    hdr = Header()
    assert_equal(hdr.get_data_dtype(), np.dtype(np.float32))
    hdr.set_data_dtype(np.float64)
    assert_equal(hdr.get_data_dtype(), np.dtype(np.float64))
    hdr.set_data_dtype('u2')
    assert_equal(hdr.get_data_dtype(), np.dtype(np.uint16))


def test_affine():
    hdr = Header(np.float64, shape=(1,2,3), zooms=(3.0, 2.0, 1.0))
    assert_array_almost_equal(hdr.get_default_affine(),
                                    [[-3.0,0,0,0],
                                     [0,2,0,-1],
                                     [0,0,1,-1],
                                     [0,0,0,1]])
    hdr.default_x_flip = False
    assert_array_almost_equal(hdr.get_default_affine(),
                                    [[3.0,0,0,0],
                                     [0,2,0,-1],
                                     [0,0,1,-1],
                                     [0,0,0,1]])
    assert_array_equal(hdr.get_base_affine(),
                             hdr.get_default_affine())


def test_read_data():
    hdr = Header(np.int32, shape=(1,2,3), zooms=(3.0, 2.0, 1.0))
    fobj = BytesIO()
    data = np.arange(6).reshape((1,2,3))
    hdr.data_to_fileobj(data, fobj)
    assert_equal(fobj.getvalue(),
                       data.astype(np.int32).tostring(order='F'))
    fobj.seek(0)
    data2 = hdr.data_from_fileobj(fobj)
    assert_array_equal(data, data2)


class DataLike(object):
    # Minimal class implementing 'data' API
    shape = (3,)
    def __array__(self):
        return np.arange(3)


class TestSpatialImage(TestCase):
    # class for testing images
    image_class = SpatialImage

    def test_isolation(self):
        # Test image isolated from external changes to header and affine
        img_klass = self.image_class
        arr = np.arange(3, dtype=np.int16)
        aff = np.eye(4)
        img = img_klass(arr, aff)
        assert_array_equal(img.get_affine(), aff)
        aff[0,0] = 99
        assert_false(np.all(img.get_affine() == aff))
        # header, created by image creation
        ihdr = img.get_header()
        # Pass it back in
        img = img_klass(arr, aff, ihdr)
        # Check modifying header outside does not modify image
        ihdr.set_zooms((4,))
        assert_not_equal(img.get_header(), ihdr)

    def test_float_affine(self):
        # Check affines get converted to float
        img_klass = self.image_class
        arr = np.arange(3, dtype=np.int16)
        img = img_klass(arr, np.eye(4, dtype=np.float32))
        assert_equal(img.get_affine().dtype, np.dtype(np.float64))
        img = img_klass(arr, np.eye(4, dtype=np.int16))
        assert_equal(img.get_affine().dtype, np.dtype(np.float64))

    def test_images(self):
        # Assumes all possible images support int16
        # See https://github.com/nipy/nibabel/issues/58
        arr = np.arange(3, dtype=np.int16)
        img = self.image_class(arr, None)
        assert_array_equal(img.get_data(), arr)
        assert_equal(img.get_affine(), None)
        hdr = self.image_class.header_class()
        hdr.set_data_shape(arr.shape)
        hdr.set_data_dtype(arr.dtype)
        assert_equal(img.get_header(), hdr)

    def test_data_api(self):
        # Test minimal api data object can initialize
        img = self.image_class(DataLike(), None)
        assert_array_equal(img.get_data(), np.arange(3))
        assert_equal(img.shape, (3,))

    def test_data_default(self):
        # check that the default dtype comes from the data if the header
        # is None, and that unsupported dtypes raise an error
        img_klass = self.image_class
        hdr_klass = self.image_class.header_class
        data = np.arange(24, dtype=np.int32).reshape((2,3,4))
        affine = np.eye(4)
        img = img_klass(data, affine)
        assert_equal(data.dtype, img.get_data_dtype())
        header = hdr_klass()
        img = img_klass(data, affine, header)
        assert_equal(img.get_data_dtype(), np.dtype(np.float32))

    def test_data_shape(self):
        # Check shape correctly read
        img_klass = self.image_class
        # Assumes all possible images support int16
        # See https://github.com/nipy/nibabel/issues/58
        arr = np.arange(4, dtype=np.int16)
        img = img_klass(arr, np.eye(4))
        assert_equal(img.shape, (4,))
        img = img_klass(np.zeros((2,3,4)), np.eye(4))
        assert_equal(img.shape, (2,3,4))

    def test_str(self):
        # Check something comes back from string representation
        img_klass = self.image_class
        # Assumes all possible images support int16
        # See https://github.com/nipy/nibabel/issues/58
        arr = np.arange(5, dtype=np.int16)
        img = img_klass(arr, np.eye(4))
        assert_true(len(str(img)) > 0)
        assert_equal(img.shape, (5,))
        img = img_klass(np.zeros((2,3,4), dtype=np.int16), np.eye(4))
        assert_true(len(str(img)) > 0)

    def test_get_shape(self):
        # Check there is a get_shape method
        # (it is deprecated)
        img_klass = self.image_class
        # Assumes all possible images support int16
        # See https://github.com/nipy/nibabel/issues/58
        img = img_klass(np.arange(1, dtype=np.int16), np.eye(4))
        assert_equal(img.get_shape(), (1,))
        img = img_klass(np.zeros((2,3,4), np.int16), np.eye(4))
        assert_equal(img.get_shape(), (2,3,4))
