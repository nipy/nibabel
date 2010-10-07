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
from StringIO import StringIO

import numpy as np

from ..spatialimages import Header, SpatialImage, \
    HeaderDataError, ImageDataError, ImageFileError

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_not_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from ..testing import data_path, parametric, \
    ParametricTestCase


@parametric
def test_header_init():
    # test the basic header
    hdr = Header()
    yield assert_equal(hdr.get_data_dtype(), np.dtype(np.float32))
    yield assert_equal(hdr.get_data_shape(), (0,))
    yield assert_equal(hdr.get_zooms(), (1.0,))
    hdr = Header(np.float64)
    yield assert_equal(hdr.get_data_dtype(), np.dtype(np.float64))
    yield assert_equal(hdr.get_data_shape(), (0,))
    yield assert_equal(hdr.get_zooms(), (1.0,))
    hdr = Header(np.float64, shape=(1,2,3))
    yield assert_equal(hdr.get_data_dtype(), np.dtype(np.float64))
    yield assert_equal(hdr.get_data_shape(), (1,2,3))
    yield assert_equal(hdr.get_zooms(), (1.0, 1.0, 1.0))
    hdr = Header(np.float64, shape=(1,2,3), zooms=None)
    yield assert_equal(hdr.get_data_dtype(), np.dtype(np.float64))
    yield assert_equal(hdr.get_data_shape(), (1,2,3))
    yield assert_equal(hdr.get_zooms(), (1.0, 1.0, 1.0))
    hdr = Header(np.float64, shape=(1,2,3), zooms=(3.0, 2.0, 1.0))
    yield assert_equal(hdr.get_data_dtype(), np.dtype(np.float64))
    yield assert_equal(hdr.get_data_shape(), (1,2,3))
    yield assert_equal(hdr.get_zooms(), (3.0, 2.0, 1.0))


@parametric
def test_from_header():
    # check from header class method.  Note equality checks below,
    # equality methods used here too.
    empty = Header.from_header()
    yield assert_equal(Header(), empty)
    empty = Header.from_header(None)
    yield assert_equal(Header(), empty)
    hdr = Header(np.float64, shape=(1,2,3), zooms=(3.0, 2.0, 1.0))
    copy = Header.from_header(hdr)
    yield assert_equal(hdr, copy)
    yield assert_false(hdr is copy)
    class C(object):
        def get_data_dtype(self): return np.dtype('u2')
        def get_data_shape(self): return (5,4,3)
        def get_zooms(self): return (10.0, 9.0, 8.0)
    converted = Header.from_header(C())
    yield assert_true(isinstance(converted, Header))
    yield assert_equal(converted.get_data_dtype(), np.dtype('u2'))
    yield assert_equal(converted.get_data_shape(), (5,4,3))
    yield assert_equal(converted.get_zooms(), (10.0,9.0,8.0))
    

@parametric
def test_eq():
    hdr = Header()
    other = Header()
    yield assert_equal(hdr, other)
    other = Header('u2')
    yield assert_not_equal(hdr, other)
    other = Header(shape=(1,2,3))
    yield assert_not_equal(hdr, other)
    hdr = Header(shape=(1,2))
    other = Header(shape=(1,2))
    yield assert_equal(hdr, other)
    other = Header(shape=(1,2), zooms=(2.0,3.0))
    yield assert_not_equal(hdr, other)
    

@parametric
def test_copy():
    # test that copy makes independent copy
    hdr = Header(np.float64, shape=(1,2,3), zooms=(3.0, 2.0, 1.0))
    hdr_copy = hdr.copy()
    hdr.set_data_shape((4,5,6))
    yield assert_equal(hdr.get_data_shape(), (4,5,6))
    yield assert_equal(hdr_copy.get_data_shape(), (1,2,3))
    hdr.set_zooms((4,5,6))
    yield assert_equal(hdr.get_zooms(), (4,5,6))
    yield assert_equal(hdr_copy.get_zooms(), (3,2,1))
    hdr.set_data_dtype(np.uint8)
    yield assert_equal(hdr.get_data_dtype(), np.dtype(np.uint8))
    yield assert_equal(hdr_copy.get_data_dtype(), np.dtype(np.float64))
    

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
    yield assert_raises(HeaderDataError,
                        hdr.set_zooms,
                        (4.0, 3.0, 2.0, 1.0))
    # as do negative zooms
    yield assert_raises(HeaderDataError,
                        hdr.set_zooms,
                        (4.0, 3.0, -2.0))
    

@parametric
def test_data_dtype():
    hdr = Header()
    yield assert_equal(hdr.get_data_dtype(), np.dtype(np.float32))
    hdr.set_data_dtype(np.float64)
    yield assert_equal(hdr.get_data_dtype(), np.dtype(np.float64))
    hdr.set_data_dtype('u2')
    yield assert_equal(hdr.get_data_dtype(), np.dtype(np.uint16))


@parametric
def test_affine():
    hdr = Header(np.float64, shape=(1,2,3), zooms=(3.0, 2.0, 1.0))
    yield assert_array_almost_equal(hdr.get_default_affine(),
                                    [[-3.0,0,0,0],
                                     [0,2,0,-1],
                                     [0,0,1,-1],
                                     [0,0,0,1]])
    hdr.default_x_flip = False
    yield assert_array_almost_equal(hdr.get_default_affine(),
                                    [[3.0,0,0,0],
                                     [0,2,0,-1],
                                     [0,0,1,-1],
                                     [0,0,0,1]])
    yield assert_array_equal(hdr.get_base_affine(),
                             hdr.get_default_affine())


@parametric
def test_read_data():
    hdr = Header(np.int32, shape=(1,2,3), zooms=(3.0, 2.0, 1.0))
    fobj = StringIO()
    data = np.arange(6).reshape((1,2,3))
    hdr.data_to_fileobj(data, fobj)
    yield assert_equal(fobj.getvalue(),
                       data.astype(np.int32).tostring(order='F'))
    fobj.seek(0)
    data2 = hdr.data_from_fileobj(fobj)
    yield assert_array_equal(data, data2)


class TestSpatialImage(ParametricTestCase):
    # class for testing images
    image_class = SpatialImage
    
    def test_images(self):
        img = self.image_class(None, None)
        yield assert_raises(ImageDataError, img.get_data)
        yield assert_equal(img.get_affine(), None)
        yield assert_equal(img.get_header(),
                           self.image_class.header_class())

    def test_data_default(self):
        # check that the default dtype comes from the data if the header
        # is None, and that unsupported dtypes raise an error
        img_klass = self.image_class
        hdr_klass = self.image_class.header_class
        data = np.arange(24, dtype=np.int32).reshape((2,3,4))
        affine = np.eye(4)
        img = img_klass(data, affine)
        yield assert_equal(data.dtype, img.get_data_dtype())
        header = hdr_klass()
        img = img_klass(data, affine, header)
        yield assert_equal(img.get_data_dtype(), np.dtype(np.float32))
