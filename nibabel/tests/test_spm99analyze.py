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
from ..casting import type_info

from ..testing import (assert_equal, assert_true, assert_false, assert_raises)

from . import test_analyze


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

    def test_big_scaling(self):
        # Test that upcasting works for huge scalefactors
        # See tests for apply_read_scaling in test_utils
        hdr = self.header_class()
        hdr.set_data_shape((1,1,1))
        hdr.set_data_dtype(np.int16)
        sio = BytesIO()
        dtt = np.float32
        # This will generate a huge scalefactor
        data = np.array([type_info(dtt)['max']], dtype=dtt)[:,None, None]
        hdr.data_to_fileobj(data, sio)
        data_back = hdr.data_from_fileobj(sio)
        assert_true(np.allclose(data, data_back))

    def test_origin_checks(self):
        HC = self.header_class
        # origin
        hdr = HC()
        hdr.data_shape = [1,1,1]
        hdr['origin'][0] = 101 # severity 20
        fhdr, message, raiser = self.log_chk(hdr, 20)
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
        fhdr, message, raiser = self.log_chk(hdr, 30)
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

    test_header_updating = (scipy_skip(
        test_analyze.TestAnalyzeImage.test_header_updating
    ))

    @scipy_skip
    def test_mat_read(self):
        # Test mat file reading and writing for the SPM analyze types
        img_klass = self.image_class
        arr = np.arange(24, dtype=np.int32).reshape((2,3,4))
        aff = np.diag([2,3,4,1]) # no LR flip in affine
        img = img_klass(arr, aff)
        fm = img.file_map
        for key, value in fm.items():
            value.fileobj = BytesIO()
        # Test round trip
        img.to_file_map()
        r_img = img_klass.from_file_map(fm)
        assert_array_equal(r_img.get_data(), arr)
        assert_array_equal(r_img.get_affine(), aff)
        # mat files are for matlab and have 111 voxel origins.  We need to
        # adjust for that, when loading and saving.  Check for signs of that in
        # the saved mat file
        mat_fileobj = img.file_map['mat'].fileobj
        from scipy.io import loadmat, savemat
        mat_fileobj.seek(0)
        mats = loadmat(mat_fileobj)
        assert_true('M' in mats and 'mat' in mats)
        from_111 = np.eye(4)
        from_111[:3,3] = -1
        to_111 = np.eye(4)
        to_111[:3,3] = 1
        assert_array_equal(mats['mat'], np.dot(aff, from_111))
        # The M matrix does not include flips, so if we only
        # have the M matrix in the mat file, and we have default flipping, the
        # mat resulting should have a flip.  The 'mat' matrix does include flips
        # and so should be unaffected by the flipping.  If both are present we
        # prefer the the 'mat' matrix.
        assert_true(img.get_header().default_x_flip) # check the default
        flipper = np.diag([-1,1,1,1])
        assert_array_equal(mats['M'], np.dot(aff, np.dot(flipper, from_111)))
        mat_fileobj.seek(0)
        savemat(mat_fileobj, dict(M=np.diag([3,4,5,1]), mat=np.diag([6,7,8,1])))
        # Check we are preferring the 'mat' matrix
        r_img = img_klass.from_file_map(fm)
        assert_array_equal(r_img.get_data(), arr)
        assert_array_equal(r_img.get_affine(),
                           np.dot(np.diag([6,7,8,1]), to_111))
        # But will use M if present
        mat_fileobj.seek(0)
        mat_fileobj.truncate(0)
        savemat(mat_fileobj, dict(M=np.diag([3,4,5,1])))
        r_img = img_klass.from_file_map(fm)
        assert_array_equal(r_img.get_data(), arr)
        assert_array_equal(r_img.get_affine(),
                           np.dot(np.diag([3,4,5,1]), np.dot(flipper, to_111)))

    def test_none_affine(self):
        # Allow for possibility of no affine resulting in nothing written into
        # mat file.  If the mat file is a filename, we just get no file, but if
        # it's a fileobj, we get an empty fileobj
        img_klass = self.image_class
        # With a None affine - no matfile written
        img = img_klass(np.zeros((2,3,4)), None)
        aff = img.get_header().get_best_affine()
        # Save / reload using bytes IO objects
        for key, value in img.file_map.items():
            value.fileobj = BytesIO()
        img.to_file_map()
        img_back = img.from_file_map(img.file_map)
        assert_array_equal(img_back.get_affine(), aff)


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

