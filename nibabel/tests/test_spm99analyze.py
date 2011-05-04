# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np

from ..py3k import BytesIO

from numpy.testing import assert_array_equal, assert_array_almost_equal, dec

# Decorator to skip tests requiring save / load if scipy not available for mat
# files
try:
    import scipy
except ImportError:
    have_scipy = False
else:
    have_scipy = True
scipy_skip = dec.skipif(not have_scipy, 'scipy not available')

from ..spm99analyze import (Spm99AnalyzeHeader, Spm99AnalyzeImage,
                            HeaderTypeError)

from ..testing import (assert_equal, assert_true, assert_false, assert_raises)

from . import test_analyze
from .test_analyze import _log_chk


class TestSpm99AnalyzeHeader(test_analyze.TestAnalyzeHeader):
    header_class = Spm99AnalyzeHeader

    def test_empty(self):
        super(TestSpm99AnalyzeHeader, self).test_empty()
        hdr = self.header_class()
        assert_equal(hdr['scl_slope'], 1)

    def test_scaling(self):
        hdr = self.header_class()
        hdr.set_data_shape((1,2,3))
        hdr.set_data_dtype(np.int16)
        S3 = BytesIO()
        data = np.arange(6, dtype=np.float64).reshape((1,2,3))
        # This uses scaling
        hdr.data_to_fileobj(data, S3)
        data_back = hdr.data_from_fileobj(S3)
        assert_array_almost_equal(data, data_back, 4)
        # This is exactly the same call, just testing it works twice
        data_back2 = hdr.data_from_fileobj(S3)
        assert_array_equal(data_back, data_back2, 4)

    def test_origin_checks(self):
        HC = self.header_class
        # origin
        hdr = HC()
        hdr.data_shape = [1,1,1]
        hdr['origin'][0] = 101 # severity 20
        fhdr, message, raiser = _log_chk(hdr, 20)
        assert_equal(fhdr, hdr)
        assert_equal(message, 'very large origin values '
                           'relative to dims; leaving as set, '
                           'ignoring for affine')
        assert_raises(*raiser)
        # diagnose binary block
        dxer = self.header_class.diagnose_binaryblock
        assert_equal(dxer(hdr.binaryblock),
                           'very large origin values '
                           'relative to dims')

    def test_spm_scale_checks(self):
        # checks for scale
        hdr = self.header_class()
        hdr['scl_slope'] = np.nan
        # NaN and Inf string representation can be odd on windows, so we
        # check against the representation on this system
        fhdr, message, raiser = _log_chk(hdr, 30)
        assert_equal(fhdr['scl_slope'], 1)
        assert_equal(message, 'scale slope is %s; '
                           'should be finite; '
                           'setting scalefactor "scl_slope" to 1' %
                           np.nan)
        assert_raises(*raiser)
        dxer = self.header_class.diagnose_binaryblock
        assert_equal(dxer(hdr.binaryblock),
                           'scale slope is %s; '
                           'should be finite' % np.nan)
        hdr['scl_slope'] = np.inf
        # Inf string representation can be odd on windows
        assert_equal(dxer(hdr.binaryblock),
                           'scale slope is %s; '
                           'should be finite'
                           % np.inf)


class TestSpm99AnalyzeImage(test_analyze.TestAnalyzeImage):
    # class for testing images
    image_class = Spm99AnalyzeImage

    # Decorating the old way, before the team invented @
    test_data_hdr_cache = (scipy_skip(
        test_analyze.TestAnalyzeImage.test_data_hdr_cache
    ))


def test_origin_affine():
    hdr = Spm99AnalyzeHeader()
    aff = hdr.get_origin_affine()
    assert_array_equal(aff, hdr.get_base_affine())
    hdr.set_data_shape((3, 5, 7))
    hdr.set_zooms((3, 2, 1))
    assert_true(hdr.default_x_flip)
    assert_array_almost_equal(
        hdr.get_origin_affine(), # from center of image
        [[-3.,  0.,  0.,  3.],
         [ 0.,  2.,  0., -4.],
         [ 0.,  0.,  1., -3.],
         [ 0.,  0.,  0.,  1.]])
    hdr['origin'][:3] = [3,4,5]
    assert_array_almost_equal(
        hdr.get_origin_affine(), # using origin
        [[-3.,  0.,  0.,  6.],
         [ 0.,  2.,  0., -6.],
         [ 0.,  0.,  1., -4.],
         [ 0.,  0.,  0.,  1.]])
    hdr['origin'] = 0 # unset origin
    hdr.set_data_shape((3, 5))
    assert_array_almost_equal(
        hdr.get_origin_affine(),
        [[-3.,  0.,  0.,  3.],
         [ 0.,  2.,  0., -4.],
         [ 0.,  0.,  1., -0.],
         [ 0.,  0.,  0.,  1.]])
    hdr.set_data_shape((3, 5, 7))
    assert_array_almost_equal(
        hdr.get_origin_affine(), # from center of image
        [[-3.,  0.,  0.,  3.],
         [ 0.,  2.,  0., -4.],
         [ 0.,  0.,  1., -3.],
         [ 0.,  0.,  0.,  1.]])


def test_slope_inter():
    hdr = Spm99AnalyzeHeader()
    assert_equal(hdr.get_slope_inter(), (1.0, None))
    for intup, outup in (((2.0,), (2.0, None)),
                         ((None,), (None, None)),
                         ((1.0, None), (1.0, None)),
                         ((0.0, None), (None, None)),
                         ((None, 0.0), (None, None))):
        hdr.set_slope_inter(*intup)
        assert_equal(hdr.get_slope_inter(), outup)
        # Check set survives through checking
        hdr = Spm99AnalyzeHeader.from_header(hdr, check=True)
        assert_equal(hdr.get_slope_inter(), outup)
    # Setting not-zero to offset raises error
    assert_raises(HeaderTypeError, hdr.set_slope_inter, None, 1.1)
    assert_raises(HeaderTypeError, hdr.set_slope_inter, 2.0, 1.1)

