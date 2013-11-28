# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
import itertools

from ..externals.six import BytesIO

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
from ..casting import type_info, shared_range
from ..volumeutils import apply_read_scaling, _dt_min_max
from ..spatialimages import supported_np_types

from nose.tools import assert_true, assert_equal, assert_raises

from ..testing import assert_allclose_safely

from . import test_analyze
from .test_helpers import bytesio_round_trip

FLOAT_TYPES = np.sctypes['float']
COMPLEX_TYPES = np.sctypes['complex']
INT_TYPES = np.sctypes['int']
UINT_TYPES = np.sctypes['uint']
CFLOAT_TYPES = FLOAT_TYPES + COMPLEX_TYPES
IUINT_TYPES = INT_TYPES + UINT_TYPES
NUMERIC_TYPES = CFLOAT_TYPES + IUINT_TYPES


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

    def test_slope_inter(self):
        hdr = self.header_class()
        assert_equal(hdr.get_slope_inter(), (1.0, None))
        for intup, outup in (((2.0,), (2.0, None)),
                            ((None,), (None, None)),
                            ((1.0, None), (1.0, None)),
                            ((0.0, None), (None, None)), # null scalings
                            ((np.nan, np.nan), (None, None)),
                            ((np.nan, None), (None, None)),
                            ((None, np.nan), (None, None)),
                            ((np.inf, None), (None, None)),
                            ((-np.inf, None), (None, None)),
                            ((None, 0.0), (None, None))):
            hdr.set_slope_inter(*intup)
            assert_equal(hdr.get_slope_inter(), outup)
            # Check set survives through checking
            hdr = Spm99AnalyzeHeader.from_header(hdr, check=True)
            assert_equal(hdr.get_slope_inter(), outup)
        # Setting not-zero to offset raises error
        assert_raises(HeaderTypeError, hdr.set_slope_inter, None, 1.1)
        assert_raises(HeaderTypeError, hdr.set_slope_inter, 2.0, 1.1)
        # Default slope is NaN
        hdr.set_slope_inter(None, None)
        assert_array_equal(hdr['scl_slope'], np.nan)

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


class ScalingMixin(object):
    # Mixin to add scaling checks to image test class

    def assert_scaling_equal(self, hdr, slope, inter):
        h_slope, h_inter = self._get_raw_scaling(hdr)
        assert_array_equal(h_slope, slope)
        assert_array_equal(h_inter, inter)

    def assert_scale_me_scaling(self, hdr):
        # Assert that header `hdr` has "scale-me" scaling
        slope, inter = self._get_raw_scaling(hdr)
        if not slope is None:
            assert_true(np.isnan(slope))
        if not inter is None:
            assert_true(np.isnan(inter))

    def _get_raw_scaling(self, hdr):
        return hdr['scl_slope'], None

    def _set_raw_scaling(self, hdr, slope, inter):
        # Brutal set of slope and inter
        hdr['scl_slope'] = slope
        if not inter is None:
            raise ValueError('inter should be None')

    def _check_write_scaling(self,
                             slope,
                             inter,
                             effective_slope,
                             effective_inter):
        # Test that explicit set of slope / inter forces write of data using
        # this slope, inter
        # We use this helper function for children of the Analyze header
        img_class = self.image_class
        arr = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
        # We're going to test rounding later
        arr[0, 0, 0] = 0.4
        arr[1, 0, 0] = 0.6
        aff = np.eye(4)
        # Implicit header gives scale-me scaling
        img = img_class(arr, aff)
        self.assert_scale_me_scaling(img.header)
        # Input header scaling reset when creating image
        hdr = img.header
        self._set_raw_scaling(hdr, slope, inter)
        img = img_class(arr, aff)
        self.assert_scale_me_scaling(img.header)
        # Array from image unchanged by scaling
        assert_array_equal(img.get_data(), arr)
        # As does round trip
        img_rt = bytesio_round_trip(img)
        self.assert_scale_me_scaling(img_rt.header)
        # Round trip array is not scaled
        assert_array_equal(img_rt.get_data(), arr)
        # Explicit scaling causes scaling after round trip
        self._set_raw_scaling(img.header, slope, inter)
        self.assert_scaling_equal(img.header, slope, inter)
        # Array from image unchanged by scaling
        assert_array_equal(img.get_data(), arr)
        # But the array scaled after round trip
        img_rt = bytesio_round_trip(img)
        assert_array_equal(img_rt.get_data(),
                           apply_read_scaling(arr,
                                              effective_slope,
                                              effective_inter))
        # The scaling set into the array proxy
        do_slope, do_inter = img.header.get_slope_inter()
        assert_array_equal(img_rt.dataobj.slope,
                           1 if do_slope is None else do_slope)
        assert_array_equal(img_rt.dataobj.inter,
                           0 if do_inter is None else do_inter)
        # The new header scaling has been reset
        self.assert_scale_me_scaling(img_rt.header)
        # But the original is the same as it was when we set it
        self.assert_scaling_equal(img.header, slope, inter)
        # The data gets rounded nicely if we need to do conversion
        img.header.set_data_dtype(np.uint8)
        img_rt = bytesio_round_trip(img)
        assert_array_equal(img_rt.get_data(),
                           apply_read_scaling(np.round(arr),
                                              effective_slope,
                                              effective_inter))
        # But we have to clip too
        arr[-1, -1, -1] = 256
        arr[-2, -1, -1] = -1
        img_rt = bytesio_round_trip(img)
        exp_unscaled_arr = np.clip(np.round(arr), 0, 255)
        assert_array_equal(img_rt.get_data(),
                           apply_read_scaling(exp_unscaled_arr,
                                              effective_slope,
                                              effective_inter))

    def test_int_int_scaling(self):
        # Check int to int conversion without slope, inter
        img_class = self.image_class
        arr = np.array([-1, 0, 256], dtype=np.int16)[:, None, None]
        img = img_class(arr, np.eye(4))
        hdr = img.header
        img.set_data_dtype(np.uint8)
        self._set_raw_scaling(hdr, 1, 0 if hdr.has_data_intercept else None)
        img_rt = bytesio_round_trip(img)
        assert_array_equal(img_rt.get_data(), np.clip(arr, 0, 255))

    def test_no_scaling(self):
        # Test writing image converting types when no scaling
        img_class = self.image_class
        hdr_class = img_class.header_class
        hdr = hdr_class()
        supported_types = supported_np_types(hdr)
        slope = 2
        inter = 10 if hdr.has_data_intercept else 0
        for in_dtype, out_dtype in itertools.product(
            FLOAT_TYPES + IUINT_TYPES,
            supported_types):
            # Need to check complex scaling
            mn_in, mx_in = _dt_min_max(in_dtype)
            arr = np.array([mn_in, -1, 0, 1, 10, mx_in], dtype=in_dtype)
            arr = arr[:, None, None] # To 3D for no good reason
            img = img_class(arr, np.eye(4), hdr)
            img.set_data_dtype(out_dtype)
            img.header.set_slope_inter(slope, inter)
            rt_img = bytesio_round_trip(img)
            back_arr = rt_img.get_data()
            exp_back = arr.copy()
            if in_dtype not in COMPLEX_TYPES:
                exp_back = arr.astype(float)
            if out_dtype in IUINT_TYPES:
                exp_back = np.round(exp_back)
                exp_back = np.clip(exp_back, *shared_range(float, out_dtype))
                exp_back = exp_back.astype(out_dtype).astype(float)
            else:
                exp_back = exp_back.astype(out_dtype)
            # Allow for small differences in large numbers
            assert_allclose_safely(back_arr,
                                   exp_back * slope + inter)

    def test_write_scaling(self):
        # Check writes with scaling set
        for slope, inter, e_slope, e_inter in (
            (1, None, 1, None),
            (0, None, 1, None),
            (np.inf, None, 1, None),
            (2, None, 2, None),
        ):
            self._check_write_scaling(slope, inter, e_slope, e_inter)


class TestSpm99AnalyzeImage(test_analyze.TestAnalyzeImage, ScalingMixin):
    # class for testing images
    image_class = Spm99AnalyzeImage

    # Decorating the old way, before the team invented @
    test_data_hdr_cache = (scipy_skip(
        test_analyze.TestAnalyzeImage.test_data_hdr_cache
    ))

    test_header_updating = (scipy_skip(
        test_analyze.TestAnalyzeImage.test_header_updating
    ))

    test_offset_to_zero = (scipy_skip(
        test_analyze.TestAnalyzeImage.test_offset_to_zero
    ))

    test_big_offset_exts = (scipy_skip(
        test_analyze.TestAnalyzeImage.test_big_offset_exts
    ))

    test_header_scaling = (scipy_skip(
        test_analyze.TestAnalyzeImage.test_header_scaling
    ))

    test_int_int_scaling = scipy_skip(ScalingMixin.test_int_int_scaling)

    test_write_scaling = scipy_skip(ScalingMixin.test_write_scaling)

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
