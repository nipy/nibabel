# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Tests for nifti reading package '''
from __future__ import division, print_function, absolute_import
import os
import warnings

import numpy as np

from ..externals.six import BytesIO
from ..casting import type_info, have_binary128
from ..tmpdirs import InTemporaryDirectory
from ..spatialimages import HeaderDataError
from ..eulerangles import euler2mat
from ..affines import from_matvec
from .. import nifti1 as nifti1
from ..nifti1 import (load, Nifti1Header, Nifti1PairHeader, Nifti1Image,
                      Nifti1Pair, Nifti1Extension, Nifti1Extensions,
                      data_type_codes, extension_codes, slice_order_codes)

from .test_arraywriters import rt_err_estimate, IUINT_TYPES
from .test_helpers import bytesio_round_trip

from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal)
from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_raises)
from nose import SkipTest

from ..testing import data_path

from . import test_analyze as tana

header_file = os.path.join(data_path, 'nifti1.hdr')
image_file = os.path.join(data_path, 'example4d.nii.gz')


# Example transformation matrix
R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]] # rotation matrix
Z = [2.0, 3.0, 4.0] # zooms
T = [20, 30, 40] # translations
A = np.eye(4)
A[:3,:3] = np.array(R) * Z # broadcasting does the job
A[:3,3] = T


class TestNifti1PairHeader(tana.TestAnalyzeHeader):
    header_class = Nifti1PairHeader
    example_file = header_file
    quat_dtype = np.float32

    def test_empty(self):
        tana.TestAnalyzeHeader.test_empty(self)
        hdr = self.header_class()
        assert_equal(hdr['magic'], hdr.pair_magic)
        assert_equal(hdr['scl_slope'], 1)
        assert_equal(hdr['vox_offset'], hdr.pair_vox_offset)

    def test_from_eg_file(self):
        hdr = self.header_class.from_fileobj(open(self.example_file, 'rb'))
        assert_equal(hdr.endianness, '<')
        assert_equal(hdr['magic'], hdr.pair_magic)
        assert_equal(hdr['sizeof_hdr'], self.sizeof_hdr)

    def test_big_scaling(self):
        # Test that upcasting works for huge scalefactors
        # See tests for apply_read_scaling in test_utils
        hdr = self.header_class()
        hdr.set_data_shape((2,1,1))
        hdr.set_data_dtype(np.int16)
        sio = BytesIO()
        dtt = np.float32
        # This will generate a huge scalefactor
        finf = type_info(dtt)
        data = np.array([finf['min'], finf['max']], dtype=dtt)[:,None, None]
        hdr.data_to_fileobj(data, sio)
        data_back = hdr.data_from_fileobj(sio)
        assert_true(np.allclose(data, data_back))

    def test_nifti_log_checks(self):
        # in addition to analyze header checks
        HC = self.header_class
        # intercept and slope
        hdr = HC()
        # Slope of 0 is OK
        hdr['scl_slope'] = 0
        fhdr, message, raiser = self.log_chk(hdr, 0)
        assert_equal((fhdr, message), (hdr, ''))
        # But not with non-zero intercept
        hdr['scl_inter'] = 3
        fhdr, message, raiser = self.log_chk(hdr, 20)
        assert_equal(fhdr['scl_inter'], 0)
        assert_equal(message,
                           'Unused "scl_inter" is 3.0; should be 0; '
                           'setting "scl_inter" to 0')
        # Or not-finite intercept
        hdr['scl_inter'] = np.nan
        # NaN string representation can be odd on windows
        nan_str = '%s' % np.nan
        fhdr, message, raiser = self.log_chk(hdr, 20)
        assert_equal(fhdr['scl_inter'], 0)
        assert_equal(message,
                           'Unused "scl_inter" is %s; should be 0; '
                           'setting "scl_inter" to 0' % nan_str)
        # Reset to usable scale
        hdr['scl_slope'] = 1
        # not finite inter is more of a problem
        hdr['scl_inter'] = np.nan # severity 30
        fhdr, message, raiser = self.log_chk(hdr, 40)
        assert_equal(fhdr['scl_inter'], 0)
        assert_equal(message,
                           '"scl_slope" is 1.0; but "scl_inter" is %s; '
                           '"scl_inter" should be finite; setting '
                           '"scl_inter" to 0' % nan_str)
        assert_raises(*raiser)
        # Not finite scale also bad, generates message for scale and offset
        hdr['scl_slope'] = np.nan
        fhdr, message, raiser = self.log_chk(hdr, 30)
        assert_equal(fhdr['scl_slope'], 0)
        assert_equal(fhdr['scl_inter'], 0)
        assert_equal(message,
                           '"scl_slope" is nan; should be finite; '
                           'Unused "scl_inter" is nan; should be 0; '
                           'setting "scl_slope" to 0 (no scaling); '
                           'setting "scl_inter" to 0')
        assert_raises(*raiser)
        # Or just scale if inter is already 0
        hdr['scl_inter'] = 0
        fhdr, message, raiser = self.log_chk(hdr, 30)
        assert_equal(fhdr['scl_slope'], 0)
        assert_equal(fhdr['scl_inter'], 0)
        assert_equal(message,
                           '"scl_slope" is nan; should be finite; '
                           'setting "scl_slope" to 0 (no scaling)')
        assert_raises(*raiser)
        # qfac
        hdr = HC()
        hdr['pixdim'][0] = 0
        fhdr, message, raiser = self.log_chk(hdr, 20)
        assert_equal(fhdr['pixdim'][0], 1)
        assert_equal(message, 'pixdim[0] (qfac) should be 1 '
                           '(default) or -1; setting qfac to 1')
        # magic and offset
        hdr = HC()
        hdr['magic'] = 'ooh'
        fhdr, message, raiser = self.log_chk(hdr, 45)
        assert_equal(fhdr['magic'], b'ooh')
        assert_equal(message, 'magic string "ooh" is not valid; '
                           'leaving as is, but future errors are likely')
        hdr['magic'] = hdr.single_magic # single file needs suitable offset
        hdr['vox_offset'] = 0
        fhdr, message, raiser = self.log_chk(hdr, 40)
        assert_equal(fhdr['vox_offset'], hdr.single_vox_offset)
        assert_equal(message, 'vox offset 0 too low for single '
                           'file nifti1; setting to minimum value '
                           'of ' + str(hdr.single_vox_offset))
        # qform, sform
        hdr = HC()
        hdr['qform_code'] = -1
        fhdr, message, raiser = self.log_chk(hdr, 30)
        assert_equal(fhdr['qform_code'], 0)
        assert_equal(message, 'qform_code -1 not valid; '
                           'setting to 0')
        hdr = HC()
        hdr['sform_code'] = -1
        fhdr, message, raiser = self.log_chk(hdr, 30)
        assert_equal(fhdr['sform_code'], 0)
        assert_equal(message, 'sform_code -1 not valid; '
                           'setting to 0')

    def test_freesurfer_hack(self):
        # For large vector images, Freesurfer appears to set dim[1] to -1 and
        # then use glmin for the vector length (an i4)
        HC = self.header_class
        # The standard case
        hdr = HC()
        hdr.set_data_shape((2, 3, 4))
        assert_equal(hdr.get_data_shape(), (2, 3, 4))
        assert_equal(hdr['glmin'], 0)
        # Just left of the freesurfer case
        dim_type = hdr.template_dtype['dim'].base
        glmin = hdr.template_dtype['glmin'].base
        too_big = int(np.iinfo(dim_type).max) + 1
        hdr.set_data_shape((too_big-1, 1, 1))
        assert_equal(hdr.get_data_shape(), (too_big-1, 1, 1))
        # The freesurfer case
        hdr.set_data_shape((too_big, 1, 1))
        assert_equal(hdr.get_data_shape(), (too_big, 1, 1))
        assert_array_equal(hdr['dim'][:4], [3, -1, 1, 1])
        assert_equal(hdr['glmin'], too_big)
        # This only works for the case of a 3D with -1, 1, 1
        assert_raises(HeaderDataError, hdr.set_data_shape, (too_big,))
        assert_raises(HeaderDataError, hdr.set_data_shape, (too_big,1))
        assert_raises(HeaderDataError, hdr.set_data_shape, (too_big,1,2))
        assert_raises(HeaderDataError, hdr.set_data_shape, (too_big,2,1))
        assert_raises(HeaderDataError, hdr.set_data_shape, (1, too_big))
        assert_raises(HeaderDataError, hdr.set_data_shape, (1, too_big, 1))
        assert_raises(HeaderDataError, hdr.set_data_shape, (1, 1, too_big))
        # Outside range of glmin raises error
        far_too_big = int(np.iinfo(glmin).max) + 1
        hdr.set_data_shape((far_too_big-1, 1, 1))
        assert_equal(hdr.get_data_shape(), (far_too_big-1, 1, 1))
        assert_raises(HeaderDataError, hdr.set_data_shape, (far_too_big,1,1))
        # glmin of zero raises error (implausible vector length)
        hdr.set_data_shape((-1,1,1))
        hdr['glmin'] = 0
        assert_raises(HeaderDataError, hdr.get_data_shape)
        # Lists or tuples or arrays will work for setting shape
        for shape in ((too_big-1, 1, 1), (too_big, 1, 1)):
            for constructor in (list, tuple, np.array):
                hdr.set_data_shape(constructor(shape))
                assert_equal(hdr.get_data_shape(), shape)

    def test_qform_sform(self):
        HC = self.header_class
        hdr = HC()
        assert_array_equal(hdr.get_qform(), np.eye(4))
        empty_sform = np.zeros((4,4))
        empty_sform[-1,-1] = 1
        assert_array_equal(hdr.get_sform(), empty_sform)
        assert_equal(hdr.get_qform(coded=True), (None, 0))
        assert_equal(hdr.get_sform(coded=True), (None, 0))
        # Affine with no shears
        nice_aff = np.diag([2, 3, 4, 1])
        # Affine with shears
        nasty_aff = from_matvec(np.arange(9).reshape((3,3)), [9, 10, 11])
        fixed_aff = unshear_44(nasty_aff)
        for in_meth, out_meth in ((hdr.set_qform, hdr.get_qform),
                                  (hdr.set_sform, hdr.get_sform)):
            in_meth(nice_aff, 2)
            aff, code = out_meth(coded=True)
            assert_array_equal(aff, nice_aff)
            assert_equal(code, 2)
            assert_array_equal(out_meth(), nice_aff) # non coded
            # Affine can also be passed if code == 0, affine will be suitably set
            in_meth(nice_aff, 0)
            assert_equal(out_meth(coded=True), (None, 0))
            assert_array_almost_equal(out_meth(), nice_aff)
            # Default qform code when previous == 0 is 2
            in_meth(nice_aff)
            aff, code = out_meth(coded=True)
            assert_equal(code, 2)
            # Unless code was non-zero before
            in_meth(nice_aff, 1)
            in_meth(nice_aff)
            aff, code = out_meth(coded=True)
            assert_equal(code, 1)
            # Can set code without modifying affine, by passing affine=None
            assert_array_equal(aff, nice_aff) # affine same as before
            in_meth(None, 3)
            aff, code = out_meth(coded=True)
            assert_array_equal(aff, nice_aff) # affine same as before
            assert_equal(code, 3)
            # affine is None on its own, or with code==0, resets code to 0
            in_meth(None, 0)
            assert_equal(out_meth(coded=True), (None, 0))
            in_meth(None)
            assert_equal(out_meth(coded=True), (None, 0))
            # List works as input
            in_meth(nice_aff.tolist())
            assert_array_equal(out_meth(), nice_aff)
        # Qform specifics
        # inexact set (with shears) is OK
        hdr.set_qform(nasty_aff, 1)
        assert_array_almost_equal(hdr.get_qform(), fixed_aff)
        # Unless allow_shears is False
        assert_raises(HeaderDataError, hdr.set_qform, nasty_aff, 1, False)
        # Reset sform, give qform a code, to test sform
        hdr.set_sform(None)
        hdr.set_qform(nice_aff, 1)
        # Check sform unchanged by setting qform
        assert_equal(hdr.get_sform(coded=True), (None, 0))
        # Setting does change the sform ouput
        hdr.set_sform(nasty_aff, 1)
        aff, code = hdr.get_sform(coded=True)
        assert_array_equal(aff, nasty_aff)
        assert_equal(code, 1)

    def test_datatypes(self):
        hdr = self.header_class()
        for code in data_type_codes.value_set():
            dt = data_type_codes.type[code]
            if dt == np.void:
                continue
            hdr.set_data_dtype(code)
            (assert_equal,
                hdr.get_data_dtype(),
                data_type_codes.dtype[code])
        # Check that checks also see new datatypes
        hdr.set_data_dtype(np.complex128)
        hdr.check_fix()

    def test_quaternion(self):
        hdr = self.header_class()
        hdr['quatern_b'] = 0
        hdr['quatern_c'] = 0
        hdr['quatern_d'] = 0
        assert_true(np.allclose(hdr.get_qform_quaternion(), [1.0, 0, 0, 0]))
        hdr['quatern_b'] = 1
        hdr['quatern_c'] = 0
        hdr['quatern_d'] = 0
        assert_true(np.allclose(hdr.get_qform_quaternion(), [0, 1, 0, 0]))
        # Check threshold set correctly for float32
        hdr['quatern_b'] = 1+np.finfo(self.quat_dtype).eps
        assert_array_almost_equal(hdr.get_qform_quaternion(), [0, 1, 0, 0])

    def test_qform(self):
        # Test roundtrip case
        ehdr = self.header_class()
        ehdr.set_qform(A)
        qA = ehdr.get_qform()
        assert_true, np.allclose(A, qA, atol=1e-5)
        assert_true, np.allclose(Z, ehdr['pixdim'][1:4])
        xfas = nifti1.xform_codes
        assert_true, ehdr['qform_code'] == xfas['aligned']
        ehdr.set_qform(A, 'scanner')
        assert_true, ehdr['qform_code'] == xfas['scanner']
        ehdr.set_qform(A, xfas['aligned'])
        assert_true, ehdr['qform_code'] == xfas['aligned']

    def test_sform(self):
        # Test roundtrip case
        ehdr = self.header_class()
        ehdr.set_sform(A)
        sA = ehdr.get_sform()
        assert_true, np.allclose(A, sA, atol=1e-5)
        xfas = nifti1.xform_codes
        assert_true, ehdr['sform_code'] == xfas['aligned']
        ehdr.set_sform(A, 'scanner')
        assert_true, ehdr['sform_code'] == xfas['scanner']
        ehdr.set_sform(A, xfas['aligned'])
        assert_true, ehdr['sform_code'] == xfas['aligned']

    def test_dim_info(self):
        ehdr = self.header_class()
        assert_true(ehdr.get_dim_info() == (None, None, None))
        for info in ((0,2,1),
                    (None, None, None),
                    (0,2,None),
                    (0,None,None),
                    (None,2,1),
                    (None, None,1),
                    ):
            ehdr.set_dim_info(*info)
            assert_true(ehdr.get_dim_info() == info)

    def test_slice_times(self):
        hdr = self.header_class()
        # error if slice dimension not specified
        assert_raises(HeaderDataError, hdr.get_slice_times)
        hdr.set_dim_info(slice=2)
        # error if slice dimension outside shape
        assert_raises(HeaderDataError, hdr.get_slice_times)
        hdr.set_data_shape((1, 1, 7))
        # error if slice duration not set
        assert_raises(HeaderDataError, hdr.get_slice_times)
        hdr.set_slice_duration(0.1)
        # We need a function to print out the Nones and floating point
        # values in a predictable way, for the tests below.
        _stringer = lambda val: val is not None and '%2.1f' % val or None
        _print_me = lambda s: list(map(_stringer, s))
        #The following examples are from the nifti1.h documentation.
        hdr['slice_code'] = slice_order_codes['sequential increasing']
        assert_equal(_print_me(hdr.get_slice_times()),
                        ['0.0', '0.1', '0.2', '0.3', '0.4',
                            '0.5', '0.6'])
        hdr['slice_start'] = 1
        hdr['slice_end'] = 5
        assert_equal(_print_me(hdr.get_slice_times()),
            [None, '0.0', '0.1', '0.2', '0.3', '0.4', None])
        hdr['slice_code'] = slice_order_codes['sequential decreasing']
        assert_equal(_print_me(hdr.get_slice_times()),
            [None, '0.4', '0.3', '0.2', '0.1', '0.0', None])
        hdr['slice_code'] = slice_order_codes['alternating increasing']
        assert_equal(_print_me(hdr.get_slice_times()),
            [None, '0.0', '0.3', '0.1', '0.4', '0.2', None])
        hdr['slice_code'] = slice_order_codes['alternating decreasing']
        assert_equal(_print_me(hdr.get_slice_times()),
            [None, '0.2', '0.4', '0.1', '0.3', '0.0', None])
        hdr['slice_code'] = slice_order_codes['alternating increasing 2']
        assert_equal(_print_me(hdr.get_slice_times()),
            [None, '0.2', '0.0', '0.3', '0.1', '0.4', None])
        hdr['slice_code'] = slice_order_codes['alternating decreasing 2']
        assert_equal(_print_me(hdr.get_slice_times()),
            [None, '0.4', '0.1', '0.3', '0.0', '0.2', None])
        # test set
        hdr = self.header_class()
        hdr.set_dim_info(slice=2)
        # need slice dim to correspond with shape
        times = [None, 0.2, 0.4, 0.1, 0.3, 0.0, None]
        assert_raises(HeaderDataError, hdr.set_slice_times, times)
        hdr.set_data_shape([1, 1, 7])
        assert_raises(HeaderDataError,
                            hdr.set_slice_times,
                            times[:-1]) # wrong length
        assert_raises(HeaderDataError,
                            hdr.set_slice_times,
                            (None,) * len(times)) # all None
        n_mid_times = times[:]
        n_mid_times[3] = None
        assert_raises(HeaderDataError,
                            hdr.set_slice_times,
                            n_mid_times) # None in middle
        funny_times = times[:]
        funny_times[3] = 0.05
        assert_raises(HeaderDataError,
                            hdr.set_slice_times,
                            funny_times) # can't get single slice duration
        hdr.set_slice_times(times)
        assert_equal(hdr.get_value_label('slice_code'),
                        'alternating decreasing')
        assert_equal(hdr['slice_start'], 1)
        assert_equal(hdr['slice_end'], 5)
        assert_array_almost_equal(hdr['slice_duration'], 0.1)

    def test_intents(self):
        ehdr = self.header_class()
        ehdr.set_intent('t test', (10,), name='some score')
        assert_equal(ehdr.get_intent(),
                    ('t test', (10.0,), 'some score'))
        # invalid intent name
        assert_raises(KeyError,
                    ehdr.set_intent, 'no intention')
        # too many parameters
        assert_raises(HeaderDataError,
                    ehdr.set_intent,
                    't test', (10,10))
        # too few parameters
        assert_raises(HeaderDataError,
                    ehdr.set_intent,
                    'f test', (10,))
        # check unset parameters are set to 0, and name to ''
        ehdr.set_intent('t test')
        assert_equal((ehdr['intent_p1'],
                    ehdr['intent_p2'],
                    ehdr['intent_p3']), (0,0,0))
        assert_equal(ehdr['intent_name'], b'')
        ehdr.set_intent('t test', (10,))
        assert_equal((ehdr['intent_p2'],
                    ehdr['intent_p3']), (0,0))

    def test_set_slice_times(self):
        hdr = self.header_class()
        hdr.set_dim_info(slice=2)
        hdr.set_data_shape([1, 1, 7])
        hdr.set_slice_duration(0.1)
        times = [0] * 6
        assert_raises(HeaderDataError, hdr.set_slice_times, times)
        times = [None] * 7
        assert_raises(HeaderDataError, hdr.set_slice_times, times)
        times = [None, 0, 1, None, 3, 4, None]
        assert_raises(HeaderDataError, hdr.set_slice_times, times)
        times = [None, 0, 1, 2.1, 3, 4, None]
        assert_raises(HeaderDataError, hdr.set_slice_times, times)
        times = [None, 0, 4, 3, 2, 1, None]
        assert_raises(HeaderDataError, hdr.set_slice_times, times)
        times = [0, 1, 2, 3, 4, 5, 6]
        hdr.set_slice_times(times)
        assert_equal(hdr['slice_code'], 1)
        assert_equal(hdr['slice_start'], 0)
        assert_equal(hdr['slice_end'], 6)
        assert_equal(hdr['slice_duration'], 1.0)
        times = [None, 0, 1, 2, 3, 4, None]
        hdr.set_slice_times(times)
        assert_equal(hdr['slice_code'], 1)
        assert_equal(hdr['slice_start'], 1)
        assert_equal(hdr['slice_end'], 5)
        assert_equal(hdr['slice_duration'], 1.0)
        times = [None, 0.4, 0.3, 0.2, 0.1, 0, None]
        hdr.set_slice_times(times)
        assert_true(np.allclose(hdr['slice_duration'], 0.1))
        times = [None, 4, 3, 2, 1, 0, None]
        hdr.set_slice_times(times)
        assert_equal(hdr['slice_code'], 2)
        times = [None, 0, 3, 1, 4, 2, None]
        hdr.set_slice_times(times)
        assert_equal(hdr['slice_code'], 3)
        times = [None, 2, 4, 1, 3, 0, None]
        hdr.set_slice_times(times)
        assert_equal(hdr['slice_code'], 4)
        times = [None, 2, 0, 3, 1, 4, None]
        hdr.set_slice_times(times)
        assert_equal(hdr['slice_code'], 5)
        times = [None, 4, 1, 3, 0, 2, None]
        hdr.set_slice_times(times)
        assert_equal(hdr['slice_code'], 6)

    def test_slope_inter(self):
        hdr = self.header_class()
        assert_equal(hdr.get_slope_inter(), (1.0, 0.0))
        for intup, outup in (((2.0,), (2.0, 0.0)),
                            ((None,), (None, None)),
                            ((3.0, None), (3.0, 0.0)),
                            ((0.0, None), (None, None)),
                            ((None, 0.0), (None, None)),
                            ((None, 3.0), (None, None)),
                            ((2.0, 3.0), (2.0, 3.0))):
            hdr.set_slope_inter(*intup)
            assert_equal(hdr.get_slope_inter(), outup)
            # Check set survives through checking
            hdr = self.header_class.from_header(hdr, check=True)
            assert_equal(hdr.get_slope_inter(), outup)

    def test_xyzt_units(self):
        hdr = self.header_class()
        assert_equal(hdr.get_xyzt_units(), ('unknown', 'unknown'))
        hdr.set_xyzt_units('mm', 'sec')
        assert_equal(hdr.get_xyzt_units(), ('mm', 'sec'))
        hdr.set_xyzt_units()
        assert_equal(hdr.get_xyzt_units(), ('unknown', 'unknown'))

    def test_recoded_fields(self):
        hdr = self.header_class()
        assert_equal(hdr.get_value_label('qform_code'), 'unknown')
        hdr['qform_code'] = 3
        assert_equal(hdr.get_value_label('qform_code'), 'talairach')
        assert_equal(hdr.get_value_label('sform_code'), 'unknown')
        hdr['sform_code'] = 3
        assert_equal(hdr.get_value_label('sform_code'), 'talairach')
        assert_equal(hdr.get_value_label('intent_code'), 'none')
        hdr.set_intent('t test', (10,), name='some score')
        assert_equal(hdr.get_value_label('intent_code'), 't test')
        assert_equal(hdr.get_value_label('slice_code'), 'unknown')
        hdr['slice_code'] = 4 # alternating decreasing
        assert_equal(hdr.get_value_label('slice_code'),
                        'alternating decreasing')


def unshear_44(affine):
    RZS = affine[:3, :3]
    zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
    R = RZS / zooms
    P, S, Qs = np.linalg.svd(R)
    PR = np.dot(P, Qs)
    return from_matvec(PR * zooms, affine[:3,3])


class TestNifti1SingleHeader(TestNifti1PairHeader):

    header_class = Nifti1Header

    def test_empty(self):
        tana.TestAnalyzeHeader.test_empty(self)
        hdr = self.header_class()
        assert_equal(hdr['magic'], hdr.single_magic)
        assert_equal(hdr['scl_slope'], 1)
        assert_equal(hdr['vox_offset'], hdr.single_vox_offset)

    def test_binblock_is_file(self):
        # Override test that binary string is the same as the file on disk; in
        # the case of the single file version of the header, we need to append
        # the extension string (4 0s)
        hdr = self.header_class()
        str_io = BytesIO()
        hdr.write_to(str_io)
        assert_equal(str_io.getvalue(), hdr.binaryblock + b'\x00' * 4)

    def test_float128(self):
        hdr = self.header_class()
        if have_binary128():
            hdr.set_data_dtype(np.longdouble)
            assert_equal(hdr.get_data_dtype().type, np.longdouble)
        else:
            assert_raises(HeaderDataError, hdr.set_data_dtype, np.longdouble)


class TestNifti1Image(tana.TestAnalyzeImage):
    # Run analyze-flavor spatialimage tests
    image_class = Nifti1Image

    def test_none_qsform(self):
        # Check that affine gets set to q/sform if header is None
        img_klass = self.image_class
        hdr_klass = img_klass.header_class
        shape = (2, 3, 4)
        data = np.arange(24).reshape(shape)
        # With specified affine
        aff = from_matvec(euler2mat(0.1, 0.2, 0.3), [11, 12, 13])
        for hdr in (None, hdr_klass()):
            img = img_klass(data, aff, hdr)
            assert_almost_equal(img.affine, aff)
            assert_almost_equal(img.header.get_sform(), aff)
            assert_almost_equal(img.header.get_qform(), aff)
        # Even if affine is default for empty header
        hdr = hdr_klass()
        hdr.set_data_shape(shape)
        default_aff = hdr.get_best_affine()
        img = img_klass(data, default_aff, None)
        assert_almost_equal(img.header.get_sform(), default_aff)
        assert_almost_equal(img.header.get_qform(), default_aff)
        # If affine is None, s/qform not set
        img = img_klass(data, None, None)
        assert_almost_equal(img.header.get_sform(), np.diag([0, 0, 0, 1]))
        assert_almost_equal(img.header.get_qform(), np.eye(4))

    def _qform_rt(self, img):
        # Round trip image after setting qform, sform codes
        hdr = img.get_header()
        hdr['qform_code'] = 3
        hdr['sform_code'] = 4
        # Save / reload using bytes IO objects
        for key, value in img.file_map.items():
            value.fileobj = BytesIO()
        img.to_file_map()
        return img.from_file_map(img.file_map)

    def test_qform_cycle(self):
        # Qform load save cycle
        img_klass = self.image_class
        # None affine
        img = img_klass(np.zeros((2,3,4)), None)
        hdr_back = self._qform_rt(img).get_header()
        assert_equal(hdr_back['qform_code'], 3)
        assert_equal(hdr_back['sform_code'], 4)
        # Try non-None affine
        img = img_klass(np.zeros((2,3,4)), np.eye(4))
        hdr_back = self._qform_rt(img).get_header()
        assert_equal(hdr_back['qform_code'], 3)
        assert_equal(hdr_back['sform_code'], 4)
        # Modify affine in-place - does it hold?
        img.get_affine()[0,0] = 9
        img.to_file_map()
        img_back = img.from_file_map(img.file_map)
        exp_aff = np.diag([9,1,1,1])
        assert_array_equal(img_back.get_affine(), exp_aff)
        hdr_back = img.get_header()
        assert_array_equal(hdr_back.get_sform(), exp_aff)
        assert_array_equal(hdr_back.get_qform(), exp_aff)

    def test_header_update_affine(self):
        # Test that updating occurs only if affine is not allclose
        img = self.image_class(np.zeros((2,3,4)), np.eye(4))
        hdr = img.get_header()
        aff = img.get_affine()
        aff[:] = np.diag([1.1, 1.1, 1.1, 1]) # inexact floats
        hdr.set_qform(aff, 2)
        hdr.set_sform(aff, 2)
        img.update_header()
        assert_equal(hdr['sform_code'], 2)
        assert_equal(hdr['qform_code'], 2)

    def test_set_qform(self):
        img = self.image_class(np.zeros((2,3,4)), np.diag([2.2, 3.3, 4.3, 1]))
        hdr = img.get_header()
        new_affine = np.diag([1.1, 1.1, 1.1, 1])
        # Affine is same as sform (best affine)
        assert_array_almost_equal(img.get_affine(), hdr.get_best_affine())
        # Reset affine to something different again
        aff_affine = np.diag([3.3, 4.5, 6.6, 1])
        img.get_affine()[:] = aff_affine
        assert_array_almost_equal(img.get_affine(), aff_affine)
        # Set qform using new_affine
        img.set_qform(new_affine, 1)
        assert_array_almost_equal(img.get_qform(), new_affine)
        assert_equal(hdr['qform_code'], 1)
        # Image get is same as header get
        assert_array_almost_equal(img.get_qform(), new_affine)
        # Coded version of get gets same information
        qaff, code = img.get_qform(coded=True)
        assert_equal(code, 1)
        assert_array_almost_equal(qaff, new_affine)
        # Image affine now reset to best affine (which is sform)
        assert_array_almost_equal(img.get_affine(), hdr.get_best_affine())
        # Reset image affine and try update_affine == False
        img.get_affine()[:] = aff_affine
        img.set_qform(new_affine, 1, update_affine=False)
        assert_array_almost_equal(img.get_affine(), aff_affine)
        # Clear qform using None, zooms unchanged
        assert_array_almost_equal(hdr.get_zooms(), [1.1, 1.1, 1.1])
        img.set_qform(None)
        qaff, code = img.get_qform(coded=True)
        assert_equal((qaff, code), (None, 0))
        assert_array_almost_equal(hdr.get_zooms(), [1.1, 1.1, 1.1])
        # Best affine similarly
        assert_array_almost_equal(img.get_affine(), hdr.get_best_affine())
        # If sform is not set, qform should update affine
        img.set_sform(None)
        img.set_qform(new_affine, 1)
        qaff, code = img.get_qform(coded=True)
        assert_equal(code, 1)
        assert_array_almost_equal(img.get_affine(), new_affine)
        new_affine[0, 1] = 2
        # If affine has has shear, should raise Error if strip_shears=False
        img.set_qform(new_affine, 2)
        assert_raises(HeaderDataError, img.set_qform, new_affine, 2, False)
        # Unexpected keyword raises error
        assert_raises(TypeError, img.get_qform, strange=True)
        # updating None affine, None header does not work, because None header
        # results in setting the sform to default
        img = self.image_class(np.zeros((2,3,4)), None)
        new_affine = np.eye(4)
        img.set_qform(new_affine, 2)
        assert_array_almost_equal(img.affine, img.header.get_best_affine())
        # Unless we unset the sform
        img.set_sform(None, update_affine=True)
        assert_array_almost_equal(img.affine, new_affine)

    def test_set_sform(self):
        orig_aff = np.diag([2.2, 3.3, 4.3, 1])
        img = self.image_class(np.zeros((2,3,4)), orig_aff)
        hdr = img.get_header()
        new_affine = np.diag([1.1, 1.1, 1.1, 1])
        qform_affine = np.diag([1.2, 1.2, 1.2, 1])
        # Reset image affine to something different again
        aff_affine = np.diag([3.3, 4.5, 6.6, 1])
        img.get_affine()[:] = aff_affine
        assert_array_almost_equal(img.get_affine(), aff_affine)
        # Sform, Qform codes are 'aligned',  'unknown' by default
        assert_equal((hdr['sform_code'], hdr['qform_code']), (2, 0))
        # Set sform using new_affine when qform is 0
        img.set_sform(new_affine, 1)
        assert_equal(hdr['sform_code'], 1)
        assert_array_almost_equal(hdr.get_sform(), new_affine)
        # Image get is same as header get
        assert_array_almost_equal(img.get_sform(), new_affine)
        # Coded version gives same result
        saff, code = img.get_sform(coded=True)
        assert_equal(code, 1)
        assert_array_almost_equal(saff, new_affine)
        # Because we've reset the sform with update_affine, the affine changes
        assert_array_almost_equal(img.get_affine(), hdr.get_best_affine())
        # Reset image affine and try update_affine == False
        img.get_affine()[:] = aff_affine
        img.set_sform(new_affine, 1, update_affine=False)
        assert_array_almost_equal(img.get_affine(), aff_affine)
        # zooms do not get updated when qform is 0
        assert_array_almost_equal(img.get_qform(), orig_aff)
        assert_array_almost_equal(hdr.get_zooms(), [2.2, 3.3, 4.3])
        img.set_qform(None)
        assert_array_almost_equal(hdr.get_zooms(), [2.2, 3.3, 4.3])
        # Set sform using new_affine when qform is set
        img.set_qform(qform_affine, 1)
        img.set_sform(new_affine, 1)
        saff, code = img.get_sform(coded=True)
        assert_equal(code, 1)
        assert_array_almost_equal(saff, new_affine)
        assert_array_almost_equal(img.get_affine(), new_affine)
        # zooms follow qform
        assert_array_almost_equal(hdr.get_zooms(), [1.2, 1.2, 1.2])
        # Clear sform using None, best_affine should fall back on qform
        img.set_sform(None)
        assert_equal(hdr['sform_code'], 0)
        assert_equal(hdr['qform_code'], 1)
        # Sform holds previous affine from last set
        assert_array_almost_equal(hdr.get_sform(), saff)
        # Image affine follows qform
        assert_array_almost_equal(img.get_affine(), qform_affine)
        assert_array_almost_equal(hdr.get_best_affine(), img.get_affine())
        # Unexpected keyword raises error
        assert_raises(TypeError, img.get_sform, strange=True)
        # updating None affine should also work
        img = self.image_class(np.zeros((2,3,4)), None)
        new_affine = np.eye(4)
        img.set_sform(new_affine, 2)
        assert_array_almost_equal(img.get_affine(), new_affine)

    def test_hdr_diff(self):
        # Check an offset beyond data does not raise an error
        img = self.image_class(np.zeros((2,3,4)), np.eye(4))
        ext = dict(img.files_types)['image']
        hdr = img.get_header()
        hdr['vox_offset'] = 400
        with InTemporaryDirectory():
            img.to_filename('another_file' + ext)

    def test_load_save(self):
        IC = self.image_class
        img_ext = IC.files_types[0][1]
        shape = (2, 4, 6)
        npt = np.float32
        data = np.arange(np.prod(shape), dtype=npt).reshape(shape)
        affine = np.diag([1, 2, 3, 1])
        img = IC(data, affine)
        assert_equal(img.shape, shape)
        img.set_data_dtype(npt)
        img2 = bytesio_round_trip(img)
        assert_array_equal(img2.get_data(), data)
        with InTemporaryDirectory() as tmpdir:
            for ext in ('.gz', '.bz2'):
                fname = os.path.join(tmpdir, 'test' + img_ext + ext)
                img.to_filename(fname)
                img3 = IC.load(fname)
                assert_true(isinstance(img3, img.__class__))
                assert_array_equal(img3.get_data(), data)
                assert_equal(img3.get_header(), img.get_header())
                # del to avoid windows errors of form 'The process cannot
                # access the file because it is being used'
                del img3

    def test_load_pixdims(self):
        # Make sure load preserves separate qform, pixdims, sform
        IC = self.image_class
        HC = IC.header_class
        arr = np.arange(24).reshape((2,3,4))
        qaff = np.diag([2, 3, 4, 1])
        saff = np.diag([5, 6, 7, 1])
        hdr = HC()
        hdr.set_qform(qaff)
        assert_array_equal(hdr.get_qform(), qaff)
        hdr.set_sform(saff)
        assert_array_equal(hdr.get_sform(), saff)
        simg = IC(arr, None, hdr)
        img_hdr = simg.get_header()
        # Check qform, sform, pixdims are the same
        assert_array_equal(img_hdr.get_qform(), qaff)
        assert_array_equal(img_hdr.get_sform(), saff)
        assert_array_equal(img_hdr.get_zooms(), [2,3,4])
        # Save to stringio
        re_simg = bytesio_round_trip(simg)
        assert_array_equal(re_simg.get_data(), arr)
        # Check qform, sform, pixdims are the same
        rimg_hdr = re_simg.get_header()
        assert_array_equal(rimg_hdr.get_qform(), qaff)
        assert_array_equal(rimg_hdr.get_sform(), saff)
        assert_array_equal(rimg_hdr.get_zooms(), [2,3,4])

    def test_affines_init(self):
        # Test we are doing vaguely spec-related qform things.  The 'spec' here
        # is some thoughts by Mark Jenkinson:
        # http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/qsform_brief_usage
        IC = self.image_class
        arr = np.arange(24).reshape((2,3,4))
        aff = np.diag([2, 3, 4, 1])
        # Default is sform set, qform not set
        img = IC(arr, aff)
        hdr = img.get_header()
        assert_equal(hdr['qform_code'], 0)
        assert_equal(hdr['sform_code'], 2)
        assert_array_equal(hdr.get_zooms(), [2, 3, 4])
        # This is also true for affines with header passed
        qaff = np.diag([3, 4, 5, 1])
        saff = np.diag([6, 7, 8, 1])
        hdr.set_qform(qaff, code='scanner')
        hdr.set_sform(saff, code='talairach')
        assert_array_equal(hdr.get_zooms(), [3, 4, 5])
        img = IC(arr, aff, hdr)
        new_hdr = img.get_header()
        # Again affine is sort of anonymous space
        assert_equal(new_hdr['qform_code'], 0)
        assert_equal(new_hdr['sform_code'], 2)
        assert_array_equal(new_hdr.get_sform(), aff)
        assert_array_equal(new_hdr.get_zooms(), [2, 3, 4])
        # But if no affine passed, codes and matrices stay the same
        img = IC(arr, None, hdr)
        new_hdr = img.get_header()
        assert_equal(new_hdr['qform_code'], 1) # scanner
        assert_array_equal(new_hdr.get_qform(), qaff)
        assert_equal(new_hdr['sform_code'], 3) # Still talairach
        assert_array_equal(new_hdr.get_sform(), saff)
        # Pixdims as in the original header
        assert_array_equal(new_hdr.get_zooms(), [3, 4, 5])


class TestNifti1Pair(TestNifti1Image):
    # Run analyze-flavor spatialimage tests
    image_class = Nifti1Pair


def test_extension_basics():
    raw = '123'
    ext = Nifti1Extension('comment', raw)
    assert_true(ext.get_sizeondisk() == 16)
    assert_true(ext.get_content() == raw)
    assert_true(ext.get_code() == 6)


def test_ext_eq():
    ext = Nifti1Extension('comment', '123')
    assert_true(ext == ext)
    assert_false(ext != ext)
    ext2 = Nifti1Extension('comment', '124')
    assert_false(ext == ext2)
    assert_true(ext != ext2)


def test_extension_codes():
    for k in extension_codes.keys():
        ext = Nifti1Extension(k, 'somevalue')


def test_extension_list():
    ext_c0 = Nifti1Extensions()
    ext_c1 = Nifti1Extensions()
    assert_equal(ext_c0, ext_c1)
    ext = Nifti1Extension('comment', '123')
    ext_c1.append(ext)
    assert_false(ext_c0 == ext_c1)
    ext_c0.append(ext)
    assert_true(ext_c0 == ext_c1)


def test_extension_io():
    bio = BytesIO()
    ext1 = Nifti1Extension(6, b'Extra comment')
    ext1.write_to(bio, False)
    bio.seek(0)
    ebacks = Nifti1Extensions.from_fileobj(bio, -1, False)
    assert_equal(len(ebacks), 1)
    assert_equal(ext1, ebacks[0])
    # Check the start is what we expect
    exp_dtype = np.dtype([('esize', 'i4'), ('ecode', 'i4')])
    bio.seek(0)
    buff = np.ndarray(shape=(), dtype=exp_dtype, buffer=bio.read(16))
    assert_equal(buff['esize'], 32)
    assert_equal(buff['ecode'], 6)
    # Try another extension on top
    bio.seek(32)
    ext2 = Nifti1Extension(6, b'Comment')
    ext2.write_to(bio, False)
    bio.seek(0)
    ebacks = Nifti1Extensions.from_fileobj(bio, -1, False)
    assert_equal(len(ebacks), 2)
    assert_equal(ext1, ebacks[0])
    assert_equal(ext2, ebacks[1])
    # Rewrite but deliberately setting esize wrongly
    bio.truncate(0)
    bio.seek(0)
    ext1.write_to(bio, False)
    bio.seek(0)
    start = np.zeros((1,), dtype=exp_dtype)
    start['esize'] = 24
    start['ecode'] = 6
    bio.write(start.tostring())
    bio.seek(24)
    ext2.write_to(bio, False)
    # Result should still be OK, but with a warning
    bio.seek(0)
    with warnings.catch_warnings(record=True) as warns:
        ebacks = Nifti1Extensions.from_fileobj(bio, -1, False)
        assert_equal(len(warns), 1)
        assert_equal(warns[0].category, UserWarning)
        assert_equal(len(ebacks), 2)
        assert_equal(ext1, ebacks[0])
        assert_equal(ext2, ebacks[1])


def test_nifti_extensions():
    nim = load(image_file)
    # basic checks of the available extensions
    hdr = nim.get_header()
    exts_container = hdr.extensions
    assert_equal(len(exts_container), 2)
    assert_equal(exts_container.count('comment'), 2)
    assert_equal(exts_container.count('afni'), 0)
    assert_equal(exts_container.get_codes(), [6, 6])
    assert_equal((exts_container.get_sizeondisk()) % 16, 0)
    # first extension should be short one
    assert_equal(exts_container[0].get_content(), b'extcomment1')
    # add one
    afniext = Nifti1Extension('afni', '<xml></xml>')
    exts_container.append(afniext)
    assert_true(exts_container.get_codes() == [6, 6, 4])
    assert_true(exts_container.count('comment') == 2)
    assert_true(exts_container.count('afni') == 1)
    assert_true((exts_container.get_sizeondisk()) % 16 == 0)
    # delete one
    del exts_container[1]
    assert_true(exts_container.get_codes() == [6, 4])
    assert_true(exts_container.count('comment') == 1)
    assert_true(exts_container.count('afni') == 1)


class TestNifti1General(object):
    """ Test class to test nifti1 in general

    Tests here which mix the pair and the single type, and that should only be
    run once (not for each type) because they are slow
    """
    single_class = Nifti1Image
    pair_class = Nifti1Pair
    module = nifti1
    example_file = image_file

    def test_loadsave_cycle(self):
        nim = self.module.load(self.example_file)
        # ensure we have extensions
        hdr = nim.get_header()
        exts_container = hdr.extensions
        assert_true(len(exts_container) > 0)
        # write into the air ;-)
        lnim = bytesio_round_trip(nim)
        hdr = lnim.get_header()
        lexts_container = hdr.extensions
        assert_equal(exts_container,
                    lexts_container)
        # build int16 image
        data = np.ones((2,3,4,5), dtype='int16')
        img = self.single_class(data, np.eye(4))
        hdr = img.get_header()
        assert_equal(hdr.get_data_dtype(), np.int16)
        # default should have no scaling
        assert_equal(hdr.get_slope_inter(), (1.0, 0.0))
        # set scaling
        hdr.set_slope_inter(2, 8)
        assert_equal(hdr.get_slope_inter(), (2, 8))
        # now build new image with updated header
        wnim = self.single_class(data, np.eye(4), header=hdr)
        assert_equal(wnim.get_data_dtype(), np.int16)
        assert_equal(wnim.get_header().get_slope_inter(), (2, 8))
        # write into the air again ;-)
        lnim = bytesio_round_trip(wnim)
        assert_equal(lnim.get_data_dtype(), np.int16)
        # the test below does not pass, because the slope and inter are
        # always reset from the data, by the image write
        raise SkipTest
        assert_equal(lnim.get_header().get_slope_inter(), (2, 8))

    def test_load(self):
        # test module level load.  We try to load a nii and an .img and a .hdr and
        # expect to get a nifti back of single or pair type
        arr = np.arange(24).reshape((2,3,4))
        aff = np.diag([2, 3, 4, 1])
        simg = self.single_class(arr, aff)
        pimg = self.pair_class(arr, aff)
        save = self.module.save
        load = self.module.load
        with InTemporaryDirectory():
            for img in (simg, pimg):
                save(img, 'test.nii')
                assert_array_equal(arr, load('test.nii').get_data())
                save(simg, 'test.img')
                assert_array_equal(arr, load('test.img').get_data())
                save(simg, 'test.hdr')
                assert_array_equal(arr, load('test.hdr').get_data())

    def test_float_int_min_max(self):
        # Conversion between float and int
        # Parallel test to arraywriters
        aff = np.eye(4)
        for in_dt in (np.float32, np.float64):
            finf = type_info(in_dt)
            arr = np.array([finf['min'], finf['max']], dtype=in_dt)
            for out_dt in IUINT_TYPES:
                img = self.single_class(arr, aff)
                img_back = bytesio_round_trip(img)
                arr_back_sc = img_back.get_data()
                assert_true(np.allclose(arr, arr_back_sc))

    def test_float_int_spread(self):
        # Test rounding error for spread of values
        # Parallel test to arraywriters
        powers = np.arange(-10, 10, 0.5)
        arr = np.concatenate((-10**powers, 10**powers))
        aff = np.eye(4)
        for in_dt in (np.float32, np.float64):
            arr_t = arr.astype(in_dt)
            for out_dt in IUINT_TYPES:
                img = self.single_class(arr_t, aff)
                img_back = bytesio_round_trip(img)
                arr_back_sc = img_back.get_data()
                slope, inter = img_back.get_header().get_slope_inter()
                # Get estimate for error
                max_miss = rt_err_estimate(arr_t, arr_back_sc.dtype, slope, inter)
                # Simulate allclose test with large atol
                diff = np.abs(arr_t - arr_back_sc)
                rdiff = diff / np.abs(arr_t)
                assert_true(np.all((diff <= max_miss) | (rdiff <= 1e-5)))

    def test_rt_bias(self):
        # Check for bias in round trip
        # Parallel test to arraywriters
        rng = np.random.RandomState(20111214)
        mu, std, count = 100, 10, 100
        arr = rng.normal(mu, std, size=(count,))
        eps = np.finfo(np.float32).eps
        aff = np.eye(4)
        for in_dt in (np.float32, np.float64):
            arr_t = arr.astype(in_dt)
            for out_dt in IUINT_TYPES:
                img = self.single_class(arr_t, aff)
                img_back = bytesio_round_trip(img)
                arr_back_sc = img_back.get_data()
                slope, inter = img_back.get_header().get_slope_inter()
                bias = np.mean(arr_t - arr_back_sc)
                # Get estimate for error
                max_miss = rt_err_estimate(arr_t, arr_back_sc.dtype, slope, inter)
                # Hokey use of max_miss as a std estimate
                bias_thresh = np.max([max_miss / np.sqrt(count), eps])
                assert_true(np.abs(bias) < bias_thresh)
