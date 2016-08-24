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

import warnings

import numpy as np

from ..externals.six import BytesIO
from ..spatialimages import (SpatialHeader, SpatialImage, HeaderDataError,
                             Header, ImageDataError)

from unittest import TestCase
from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_not_equal, assert_raises)
from numpy.testing import assert_array_equal, assert_array_almost_equal

from .test_helpers import bytesio_round_trip
from ..testing import (clear_and_catch_warnings, suppress_warnings,
                       VIRAL_MEMMAP)
from ..tmpdirs import InTemporaryDirectory
from .. import load as top_load


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
    hdr = Header(np.float64, shape=(1, 2, 3))
    assert_equal(hdr.get_data_dtype(), np.dtype(np.float64))
    assert_equal(hdr.get_data_shape(), (1, 2, 3))
    assert_equal(hdr.get_zooms(), (1.0, 1.0, 1.0))
    hdr = Header(np.float64, shape=(1, 2, 3), zooms=None)
    assert_equal(hdr.get_data_dtype(), np.dtype(np.float64))
    assert_equal(hdr.get_data_shape(), (1, 2, 3))
    assert_equal(hdr.get_zooms(), (1.0, 1.0, 1.0))
    hdr = Header(np.float64, shape=(1, 2, 3), zooms=(3.0, 2.0, 1.0))
    assert_equal(hdr.get_data_dtype(), np.dtype(np.float64))
    assert_equal(hdr.get_data_shape(), (1, 2, 3))
    assert_equal(hdr.get_zooms(), (3.0, 2.0, 1.0))


def test_from_header():
    # check from header class method.  Note equality checks below,
    # equality methods used here too.
    empty = Header.from_header()
    assert_equal(Header(), empty)
    empty = Header.from_header(None)
    assert_equal(Header(), empty)
    hdr = Header(np.float64, shape=(1, 2, 3), zooms=(3.0, 2.0, 1.0))
    copy = Header.from_header(hdr)
    assert_equal(hdr, copy)
    assert_false(hdr is copy)

    class C(object):

        def get_data_dtype(self): return np.dtype('u2')

        def get_data_shape(self): return (5, 4, 3)

        def get_zooms(self): return (10.0, 9.0, 8.0)
    converted = Header.from_header(C())
    assert_true(isinstance(converted, Header))
    assert_equal(converted.get_data_dtype(), np.dtype('u2'))
    assert_equal(converted.get_data_shape(), (5, 4, 3))
    assert_equal(converted.get_zooms(), (10.0, 9.0, 8.0))


def test_eq():
    hdr = Header()
    other = Header()
    assert_equal(hdr, other)
    other = Header('u2')
    assert_not_equal(hdr, other)
    other = Header(shape=(1, 2, 3))
    assert_not_equal(hdr, other)
    hdr = Header(shape=(1, 2))
    other = Header(shape=(1, 2))
    assert_equal(hdr, other)
    other = Header(shape=(1, 2), zooms=(2.0, 3.0))
    assert_not_equal(hdr, other)


def test_copy():
    # test that copy makes independent copy
    hdr = Header(np.float64, shape=(1, 2, 3), zooms=(3.0, 2.0, 1.0))
    hdr_copy = hdr.copy()
    hdr.set_data_shape((4, 5, 6))
    assert_equal(hdr.get_data_shape(), (4, 5, 6))
    assert_equal(hdr_copy.get_data_shape(), (1, 2, 3))
    hdr.set_zooms((4, 5, 6))
    assert_equal(hdr.get_zooms(), (4, 5, 6))
    assert_equal(hdr_copy.get_zooms(), (3, 2, 1))
    hdr.set_data_dtype(np.uint8)
    assert_equal(hdr.get_data_dtype(), np.dtype(np.uint8))
    assert_equal(hdr_copy.get_data_dtype(), np.dtype(np.float64))


def test_shape_zooms():
    hdr = Header()
    hdr.set_data_shape((1, 2, 3))
    assert_equal(hdr.get_data_shape(), (1, 2, 3))
    assert_equal(hdr.get_zooms(), (1.0, 1.0, 1.0))
    hdr.set_zooms((4, 3, 2))
    assert_equal(hdr.get_zooms(), (4.0, 3.0, 2.0))
    hdr.set_data_shape((1, 2))
    assert_equal(hdr.get_data_shape(), (1, 2))
    assert_equal(hdr.get_zooms(), (4.0, 3.0))
    hdr.set_data_shape((1, 2, 3))
    assert_equal(hdr.get_data_shape(), (1, 2, 3))
    assert_equal(hdr.get_zooms(), (4.0, 3.0, 1.0))
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
    hdr = Header(np.float64, shape=(1, 2, 3), zooms=(3.0, 2.0, 1.0))
    assert_array_almost_equal(hdr.get_best_affine(),
                              [[-3.0, 0, 0, 0],
                               [0, 2, 0, -1],
                               [0, 0, 1, -1],
                               [0, 0, 0, 1]])
    hdr.default_x_flip = False
    assert_array_almost_equal(hdr.get_best_affine(),
                              [[3.0, 0, 0, 0],
                               [0, 2, 0, -1],
                               [0, 0, 1, -1],
                               [0, 0, 0, 1]])
    assert_array_equal(hdr.get_base_affine(),
                       hdr.get_best_affine())


def test_read_data():
    class CHeader(SpatialHeader):
        data_layout = 'C'
    for klass, order in ((SpatialHeader, 'F'), (CHeader, 'C')):
        hdr = klass(np.int32, shape=(1, 2, 3), zooms=(3.0, 2.0, 1.0))
        fobj = BytesIO()
        data = np.arange(6).reshape((1, 2, 3))
        hdr.data_to_fileobj(data, fobj)
        assert_equal(fobj.getvalue(),
                     data.astype(np.int32).tostring(order=order))
        # data_to_fileobj accepts kwarg 'rescale', but no effect in this case
        fobj.seek(0)
        hdr.data_to_fileobj(data, fobj, rescale=True)
        assert_equal(fobj.getvalue(),
                     data.astype(np.int32).tostring(order=order))
        # data_to_fileobj can be a list
        fobj.seek(0)
        hdr.data_to_fileobj(data.tolist(), fobj, rescale=True)
        assert_equal(fobj.getvalue(),
                     data.astype(np.int32).tostring(order=order))
        # Read data back again
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
    can_save = False

    def test_isolation(self):
        # Test image isolated from external changes to header and affine
        img_klass = self.image_class
        arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        aff = np.eye(4)
        img = img_klass(arr, aff)
        assert_array_equal(img.affine, aff)
        aff[0, 0] = 99
        assert_false(np.all(img.affine == aff))
        # header, created by image creation
        ihdr = img.header
        # Pass it back in
        img = img_klass(arr, aff, ihdr)
        # Check modifying header outside does not modify image
        ihdr.set_zooms((4, 5, 6))
        assert_not_equal(img.header, ihdr)

    def test_float_affine(self):
        # Check affines get converted to float
        img_klass = self.image_class
        arr = np.arange(3, dtype=np.int16)
        img = img_klass(arr, np.eye(4, dtype=np.float32))
        assert_equal(img.affine.dtype, np.dtype(np.float64))
        img = img_klass(arr, np.eye(4, dtype=np.int16))
        assert_equal(img.affine.dtype, np.dtype(np.float64))

    def test_images(self):
        # Assumes all possible images support int16
        # See https://github.com/nipy/nibabel/issues/58
        arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        img = self.image_class(arr, None)
        assert_array_equal(img.get_data(), arr)
        assert_equal(img.affine, None)

    def test_default_header(self):
        # Check default header is as expected
        arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        img = self.image_class(arr, None)
        hdr = self.image_class.header_class()
        hdr.set_data_shape(arr.shape)
        hdr.set_data_dtype(arr.dtype)
        assert_equal(img.header, hdr)

    def test_data_api(self):
        # Test minimal api data object can initialize
        img = self.image_class(DataLike(), None)
        assert_array_equal(img.get_data(), np.arange(3))
        assert_equal(img.shape, (3,))

    def check_dtypes(self, expected, actual):
        # Some images will want dtypes to be equal including endianness,
        # others may only require the same type
        assert_equal(expected, actual)

    def test_data_default(self):
        # check that the default dtype comes from the data if the header
        # is None, and that unsupported dtypes raise an error
        img_klass = self.image_class
        hdr_klass = self.image_class.header_class
        data = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
        affine = np.eye(4)
        img = img_klass(data, affine)
        self.check_dtypes(data.dtype, img.get_data_dtype())
        header = hdr_klass()
        header.set_data_dtype(np.float32)
        img = img_klass(data, affine, header)
        self.check_dtypes(np.dtype(np.float32), img.get_data_dtype())

    def test_data_shape(self):
        # Check shape correctly read
        img_klass = self.image_class
        # Assumes all possible images support int16
        # See https://github.com/nipy/nibabel/issues/58
        arr = np.arange(4, dtype=np.int16)
        img = img_klass(arr, np.eye(4))
        assert_equal(img.shape, (4,))
        img = img_klass(np.zeros((2, 3, 4), dtype=np.float32), np.eye(4))
        assert_equal(img.shape, (2, 3, 4))

    def test_str(self):
        # Check something comes back from string representation
        img_klass = self.image_class
        # Assumes all possible images support int16
        # See https://github.com/nipy/nibabel/issues/58
        arr = np.arange(5, dtype=np.int16)
        img = img_klass(arr, np.eye(4))
        assert_true(len(str(img)) > 0)
        assert_equal(img.shape, (5,))
        img = img_klass(np.zeros((2, 3, 4), dtype=np.int16), np.eye(4))
        assert_true(len(str(img)) > 0)

    def test_get_shape(self):
        # Check there is a get_shape method
        # (it is deprecated)
        img_klass = self.image_class
        # Assumes all possible images support int16
        # See https://github.com/nipy/nibabel/issues/58
        img = img_klass(np.arange(1, dtype=np.int16), np.eye(4))
        with suppress_warnings():
            assert_equal(img.get_shape(), (1,))
            img = img_klass(np.zeros((2, 3, 4), np.int16), np.eye(4))
            assert_equal(img.get_shape(), (2, 3, 4))

    def test_get_data(self):
        # Test array image and proxy image interface
        img_klass = self.image_class
        in_data_template = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        in_data = in_data_template.copy()
        img = img_klass(in_data, None)
        assert_true(in_data is img.dataobj)
        out_data = img.get_data()
        assert_true(in_data is out_data)
        # and that uncache has no effect
        img.uncache()
        assert_true(in_data is out_data)
        assert_array_equal(out_data, in_data_template)
        # If we can save, we can create a proxy image
        if not self.can_save:
            return
        rt_img = bytesio_round_trip(img)
        assert_false(in_data is rt_img.dataobj)
        assert_array_equal(rt_img.dataobj, in_data)
        out_data = rt_img.get_data()
        assert_array_equal(out_data, in_data)
        assert_false(rt_img.dataobj is out_data)
        # cache
        assert_true(rt_img.get_data() is out_data)
        out_data[:] = 42
        rt_img.uncache()
        assert_false(rt_img.get_data() is out_data)
        assert_array_equal(rt_img.get_data(), in_data)

    def test_api_deprecations(self):

        class FakeImage(self.image_class):

            files_types = (('image', '.foo'),)

            @classmethod
            def to_file_map(self, file_map=None):
                pass

            @classmethod
            def from_file_map(self, file_map=None):
                pass

        arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        aff = np.eye(4)
        img = FakeImage(arr, aff)
        bio = BytesIO()
        file_map = FakeImage.make_file_map({'image': bio})

        with clear_and_catch_warnings() as w:
            warnings.simplefilter('always', DeprecationWarning)
            img.to_files(file_map)
            assert_equal(len(w), 1)
            img.to_filespec('an_image')
            assert_equal(len(w), 2)
            img = FakeImage.from_files(file_map)
            assert_equal(len(w), 3)
            file_map = FakeImage.filespec_to_files('an_image')
            assert_equal(list(file_map), ['image'])
            assert_equal(file_map['image'].filename, 'an_image.foo')
            assert_equal(len(w), 4)


class MmapImageMixin(object):
    """ Mixin for testing images that may return memory maps """
    #: whether to test mode of returned memory map
    check_mmap_mode = True

    def get_disk_image(self):
        """ Return image, image filename, and flag for required scaling

        Subclasses can do anything to return an image, including loading a
        pre-existing image from disk.

        Returns
        -------
        img : class:`SpatialImage` instance
        fname : str
            Image filename.
        has_scaling : bool
            True if the image array has scaling to apply to the raw image array
            data, False otherwise.
        """
        img_klass = self.image_class
        shape = (3, 4, 2)
        data = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
        img = img_klass(data, None)
        fname = 'test' + img_klass.files_types[0][1]
        img.to_filename(fname)
        return img, fname, False

    def test_load_mmap(self):
        # Test memory mapping when loading images
        img_klass = self.image_class
        with InTemporaryDirectory():
            img, fname, has_scaling = self.get_disk_image()
            file_map = img.file_map.copy()
            for func, param1 in ((img_klass.from_filename, fname),
                                 (img_klass.load, fname),
                                 (top_load, fname),
                                 (img_klass.from_file_map, file_map)):
                for mmap, expected_mode in (
                        # mmap value, expected memmap mode
                        # mmap=None -> no mmap value
                        # expected mode=None -> no memmap returned
                        (None, 'c'),
                        (True, 'c'),
                        ('c', 'c'),
                        ('r', 'r'),
                        (False, None)):
                    # If the image has scaling, then numpy 1.12 will not return
                    # a memmap, regardless of the input flags.  Previous
                    # numpies returned a memmap object, even though the array
                    # has no mmap memory backing.  See:
                    # https://github.com/numpy/numpy/pull/7406
                    if has_scaling and not VIRAL_MEMMAP:
                        expected_mode = None
                    kwargs = {}
                    if mmap is not None:
                        kwargs['mmap'] = mmap
                    back_img = func(param1, **kwargs)
                    back_data = back_img.get_data()
                    if expected_mode is None:
                        assert_false(isinstance(back_data, np.memmap),
                                     'Should not be a %s' % img_klass.__name__)
                    else:
                        assert_true(isinstance(back_data, np.memmap),
                                    'Not a %s' % img_klass.__name__)
                        if self.check_mmap_mode:
                            assert_equal(back_data.mode, expected_mode)
                    del back_img, back_data
                # Check that mmap is keyword-only
                assert_raises(TypeError, func, param1, True)
                # Check invalid values raise error
                assert_raises(ValueError, func, param1, mmap='rw')
                assert_raises(ValueError, func, param1, mmap='r+')


def test_header_deprecated():
    with clear_and_catch_warnings() as w:
        warnings.simplefilter('always', DeprecationWarning)

        class MyHeader(Header):
            pass
        assert_equal(len(w), 0)

        MyHeader()
        assert_equal(len(w), 1)
