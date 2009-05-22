from StringIO import StringIO

import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nifti.header_ufuncs import read_unscaled_data, read_data, \
    write_data, write_scaled_data

from nifti.spm99analyze import Spm99AnalyzeHeader

from nifti.testing import assert_equal, assert_true, assert_false, \
     assert_raises

from test_analyze import TestAnalyzeHeader as _TAH

class TestSpm99AnalyzeHeader(_TAH):
    header_class = Spm99AnalyzeHeader

    def test_empty(self):
        for tests in super(TestSpm99AnalyzeHeader, self).test_empty():
            yield tests
        hdr = self.header_class()
        yield assert_equal, hdr['scl_slope'], 1
    
    def test_scaling(self):
        hdr = self.header_class()
        hdr.set_data_shape((1,2,3))
        hdr.set_data_dtype(np.int16)
        S3 = StringIO()
        data = np.arange(6, dtype=np.float64).reshape((1,2,3))
        # This uses scaling
        write_scaled_data(hdr, data, S3)
        data_back = read_data(hdr, S3)
        yield assert_array_almost_equal, data, data_back, 4
        # This is exactly the same call, just testing it works twice
        data_back2 = read_data(hdr, S3)
        yield assert_array_equal, data_back, data_back2, 4


def test_checks():
    # Test extra checks for spm99
    dxer = Spm99AnalyzeHeader.diagnose_binaryblock
    hdr = Spm99AnalyzeHeader()
    hdr.data_shape = [1,1,1]
    yield assert_equal, dxer(hdr.binaryblock), ''
    hdr['origin'][0] = 101
    yield assert_equal, dxer(hdr.binaryblock), 'very large origin values relative to dims'
    hdr['origin'] = 0
    hdr['scl_slope'] = 0
    yield assert_equal, dxer(hdr.binaryblock), 'scale slope is 0.0; should !=0 and be finite'
    hdr['scl_slope'] = np.inf
    yield assert_equal, dxer(hdr.binaryblock), 'scale slope is inf; should !=0 and be finite'
    

def test_origin_affine():
    # check that origin affine works, only
    hdr = Spm99AnalyzeHeader()
    aff = hdr.get_origin_affine()
