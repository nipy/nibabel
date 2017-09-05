""" Testing filebasedimages module
"""

from itertools import product

import numpy as np

from nibabel.filebasedimages import FileBasedHeader, FileBasedImage

from nibabel.tests.test_image_api import GenericImageAPI

from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_not_equal)


class FBNumpyImage(FileBasedImage):
    header_class = FileBasedHeader
    valid_exts = ('.npy',)
    files_types = (('image', '.npy'),)

    def __init__(self, arr, header=None, extra=None, file_map=None):
        super(FBNumpyImage, self).__init__(header, extra, file_map)
        self.arr = arr

    @property
    def shape(self):
        return self.arr.shape

    def get_data(self):
        return self.arr

    def get_fdata(self):
        return self.arr.astype(np.float64)

    @classmethod
    def from_file_map(klass, file_map):
        with file_map['image'].get_prepare_fileobj('rb') as fobj:
            arr = np.load(fobj)
        return klass(arr)

    def to_file_map(self, file_map=None):
        file_map = self.file_map if file_map is None else file_map
        with file_map['image'].get_prepare_fileobj('wb') as fobj:
            np.save(fobj, self.arr)

    def get_data_dtype(self):
        return self.arr.dtype

    def set_data_dtype(self, dtype):
        self.arr = self.arr.astype(dtype)


class TestFBImageAPI(GenericImageAPI):
    """ Validation for FileBasedImage instances
    """
    # A callable returning an image from ``image_maker(data, header)``
    image_maker = FBNumpyImage
    # A callable returning a header from ``header_maker()``
    header_maker = FileBasedHeader
    # Example shapes for created images
    example_shapes = ((2,), (2, 3), (2, 3, 4), (2, 3, 4, 5))
    example_dtypes = (np.int8, np.uint16, np.int32, np.float32)
    can_save = True
    standard_extension = '.npy'

    def make_imaker(self, arr, header=None):
        return lambda: self.image_maker(arr, header)

    def obj_params(self):
        # Create new images
        for shape, dtype in product(self.example_shapes, self.example_dtypes):
            arr = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
            hdr = self.header_maker()
            func = self.make_imaker(arr.copy(), hdr)
            params = dict(
                dtype=dtype,
                data=arr,
                shape=shape,
                is_proxy=False)
            yield func, params


def test_filebased_header():
    # Test stuff about the default FileBasedHeader

    class H(FileBasedHeader):

        def __init__(self, seq=None):
            if seq is None:
                seq = []
            self.a_list = list(seq)

    in_list = [1, 3, 2]
    hdr = H(in_list)
    hdr_c = hdr.copy()
    assert_equal(hdr_c.a_list, hdr.a_list)
    # Copy is independent of original
    hdr_c.a_list[0] = 99
    assert_not_equal(hdr_c.a_list, hdr.a_list)
    # From header does a copy
    hdr2 = H.from_header(hdr)
    assert_true(isinstance(hdr2, H))
    assert_equal(hdr2.a_list, hdr.a_list)
    hdr2.a_list[0] = 42
    assert_not_equal(hdr2.a_list, hdr.a_list)
    # Default header input to from_heder gives new empty header
    hdr3 = H.from_header()
    assert_true(isinstance(hdr3, H))
    assert_equal(hdr3.a_list, [])
    hdr4 = H.from_header(None)
    assert_true(isinstance(hdr4, H))
    assert_equal(hdr4.a_list, [])
