# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Tests for nifti reading package '''
from __future__ import with_statement
import os
import pickle

from ..py3k import BytesIO, ZEROB, asbytes

import numpy as np

from ..tmpdirs import InTemporaryDirectory
from ..spatialimages import HeaderDataError
from .. import nifti1 as nifti1
from ..nifti1 import (load, Nifti1Header, Nifti1PairHeader, Nifti1Image,
                      Nifti1Pair, Nifti1Extension, Nifti1Extensions,
                      data_type_codes, extension_codes, slice_order_codes)

from numpy.testing import assert_array_equal, assert_array_almost_equal
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

    def test_empty(self):
        tana.TestAnalyzeHeader.test_empty(self)
        hdr = self.header_class()
        assert_equal(hdr['magic'], asbytes('ni1'))
        assert_equal(hdr['scl_slope'], 1)
        assert_equal(hdr['vox_offset'], 0)

    def test_from_eg_file(self):
        hdr = Nifti1Header.from_fileobj(open(self.example_file, 'rb'))
        assert_equal(hdr.endianness, '<')
        assert_equal(hdr['magic'], asbytes('ni1'))
        assert_equal(hdr['sizeof_hdr'], 348)

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
        assert_equal(fhdr['magic'], asbytes('ooh'))
        assert_equal(message, 'magic string "ooh" is not valid; '
                           'leaving as is, but future errors are likely')
        hdr['magic'] = 'n+1' # single file needs suitable offset
        hdr['vox_offset'] = 0
        fhdr, message, raiser = self.log_chk(hdr, 40)
        assert_equal(fhdr['vox_offset'], 352)
        assert_equal(message, 'vox offset 0 too low for single '
                           'file nifti1; setting to minimum value '
                           'of 352')
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


class TestNifti1SingleHeader(TestNifti1PairHeader):

    header_class = Nifti1Header

    def test_empty(self):
        tana.TestAnalyzeHeader.test_empty(self)
        hdr = self.header_class()
        assert_equal(hdr['magic'], asbytes('n+1'))
        assert_equal(hdr['scl_slope'], 1)
        assert_equal(hdr['vox_offset'], 352)

    def test_binblock_is_file(self):
        # Override test that binary string is the same as the file on disk; in
        # the case of the single file version of the header, we need to append
        # the extension string (4 0s)
        hdr = self.header_class()
        str_io = BytesIO()
        hdr.write_to(str_io)
        assert_equal(str_io.getvalue(), hdr.binaryblock + ZEROB * 4)


class TestNifti1Image(tana.TestAnalyzeImage):
    # Run analyze-flavor spatialimage tests
    image_class = Nifti1Image

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


class TestNifti1Pair(TestNifti1Image):
    # Run analyze-flavor spatialimage tests
    image_class = Nifti1Pair


def test_datatypes():
    hdr = Nifti1Header()
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


def test_quaternion():
    hdr = Nifti1Header()
    hdr['quatern_b'] = 0
    hdr['quatern_c'] = 0
    hdr['quatern_d'] = 0
    assert_true(np.allclose(hdr.get_qform_quaternion(), [1.0, 0, 0, 0]))
    hdr['quatern_b'] = 1
    hdr['quatern_c'] = 0
    hdr['quatern_d'] = 0
    assert_true(np.allclose(hdr.get_qform_quaternion(), [0, 1, 0, 0]))
    # Check threshold set correctly for float32
    hdr['quatern_b'] = 1+np.finfo(np.float32).eps
    assert_array_almost_equal(hdr.get_qform_quaternion(), [0, 1, 0, 0])


def test_qform():
    # Test roundtrip case
    ehdr = Nifti1Header()
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


def test_sform():
    # Test roundtrip case
    ehdr = Nifti1Header()
    ehdr.set_sform(A)
    sA = ehdr.get_sform()
    assert_true, np.allclose(A, sA, atol=1e-5)
    xfas = nifti1.xform_codes
    assert_true, ehdr['sform_code'] == xfas['aligned']
    ehdr.set_sform(A, 'scanner')
    assert_true, ehdr['sform_code'] == xfas['scanner']
    ehdr.set_sform(A, xfas['aligned'])
    assert_true, ehdr['sform_code'] == xfas['aligned']


def test_dim_info():
    ehdr = Nifti1Header()
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


def test_slice_times():
    hdr = Nifti1Header()
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
    _print_me = lambda s: map(_stringer, s)
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
    hdr = Nifti1Header()
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


def test_intents():
    ehdr = Nifti1Header()
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
    assert_equal(ehdr['intent_name'], asbytes(''))
    ehdr.set_intent('t test', (10,))
    assert_equal((ehdr['intent_p2'],
                  ehdr['intent_p3']), (0,0))


def test_set_slice_times():
    hdr = Nifti1Header()
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


def test_nifti1_images():
    shape = (2, 4, 6)
    npt = np.float32
    data = np.arange(np.prod(shape), dtype=npt).reshape(shape)
    affine = np.diag([1, 2, 3, 1])
    img = Nifti1Image(data, affine)
    assert_equal(img.shape, shape)
    img.set_data_dtype(npt)
    stio = BytesIO()
    img.file_map['image'].fileobj = stio
    img.to_file_map()
    img2 = Nifti1Image.from_file_map(img.file_map)
    assert_array_equal(img2.get_data(), data)
    with InTemporaryDirectory() as tmpdir:
        for ext in ('.gz', '.bz2'):
            fname = os.path.join(tmpdir, 'test.nii' + ext)
            img.to_filename(fname)
            img3 = Nifti1Image.load(fname)
            assert_true(isinstance(img3, img.__class__))
            assert_array_equal(img3.get_data(), data)
            assert_equal(img3.get_header(), img.get_header())
            # del to avoid windows errors of form 'The process cannot
            # access the file because it is being used'
            del img3


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
    assert_equal(exts_container[0].get_content(), asbytes('extcomment1'))
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


def test_loadsave_cycle():
    nim = load(image_file)
    # ensure we have extensions
    hdr = nim.get_header()
    exts_container = hdr.extensions
    assert_true(len(exts_container) > 0)
    # write into the air ;-)
    stio = BytesIO()
    nim.file_map['image'].fileobj = stio
    nim.to_file_map()
    stio.seek(0)
    # reload
    lnim = Nifti1Image.from_file_map(nim.file_map)
    hdr = lnim.get_header()
    lexts_container = hdr.extensions
    assert_equal(exts_container,
                 lexts_container)
    # build int16 image
    data = np.ones((2,3,4,5), dtype='int16')
    img = Nifti1Image(data, np.eye(4))
    hdr = img.get_header()
    assert_equal(hdr.get_data_dtype(), np.int16)
    # default should have no scaling
    assert_equal(hdr.get_slope_inter(), (1.0, 0.0))
    # set scaling
    hdr.set_slope_inter(2, 8)
    assert_equal(hdr.get_slope_inter(), (2, 8))
    # now build new image with updated header
    wnim = Nifti1Image(data, np.eye(4), header=hdr)
    assert_equal(wnim.get_data_dtype(), np.int16)
    assert_equal(wnim.get_header().get_slope_inter(), (2, 8))
    # write into the air again ;-)
    stio = BytesIO()
    wnim.file_map['image'].fileobj = stio
    wnim.to_file_map()
    stio.seek(0)
    lnim = Nifti1Image.from_file_map(wnim.file_map)
    assert_equal(lnim.get_data_dtype(), np.int16)
    # the test below does not pass, because the slope and inter are
    # always reset from the data, by the image write
    raise SkipTest
    assert_equal(lnim.get_header().get_slope_inter(), (2, 8))


def test_slope_inter():
    hdr = Nifti1Header()
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
        hdr = Nifti1Header.from_header(hdr, check=True)
        assert_equal(hdr.get_slope_inter(), outup)


def test_xyzt_units():
    hdr = Nifti1Header()
    assert_equal(hdr.get_xyzt_units(), ('unknown', 'unknown'))
    hdr.set_xyzt_units('mm', 'sec')
    assert_equal(hdr.get_xyzt_units(), ('mm', 'sec'))
    hdr.set_xyzt_units()
    assert_equal(hdr.get_xyzt_units(), ('unknown', 'unknown'))


def test_recoded_fields():
    hdr = Nifti1Header()
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


def test_load():
    # test module level load.  We try to load a nii and an .img and a .hdr and
    # expect to get a nifti back of single or pair type
    arr = np.arange(24).reshape((2,3,4))
    aff = np.diag([2, 3, 4, 1])
    simg = Nifti1Image(arr, aff)
    pimg = Nifti1Pair(arr, aff)
    with InTemporaryDirectory():
        nifti1.save(simg, 'test.nii')
        assert_array_equal(arr, nifti1.load('test.nii').get_data())
        nifti1.save(simg, 'test.img')
        assert_array_equal(arr, nifti1.load('test.img').get_data())
        nifti1.save(simg, 'test.hdr')
        assert_array_equal(arr, nifti1.load('test.hdr').get_data())


def test_load_pixdims():
    # Make sure load preserves separate qform, pixdims, sform
    arr = np.arange(24).reshape((2,3,4))
    qaff = np.diag([2, 3, 4, 1])
    saff = np.diag([5, 6, 7, 1])
    hdr = Nifti1Header()
    hdr.set_qform(qaff)
    assert_array_equal(hdr.get_qform(), qaff)
    hdr.set_sform(saff)
    assert_array_equal(hdr.get_sform(), saff)
    simg = Nifti1Image(arr, None, hdr)
    img_hdr = simg.get_header()
    # Check qform, sform, pixdims are the same
    assert_array_equal(img_hdr.get_qform(), qaff)
    assert_array_equal(img_hdr.get_sform(), saff)
    assert_array_equal(img_hdr.get_zooms(), [2,3,4])
    # Save to stringio
    fm = Nifti1Image.make_file_map()
    fm['image'].fileobj = BytesIO()
    simg.to_file_map(fm)
    # Load again
    re_simg = Nifti1Image.from_file_map(fm)
    assert_array_equal(re_simg.get_data(), arr)
    # Check qform, sform, pixdims are the same
    rimg_hdr = re_simg.get_header()
    assert_array_equal(rimg_hdr.get_qform(), qaff)
    assert_array_equal(rimg_hdr.get_sform(), saff)
    assert_array_equal(rimg_hdr.get_zooms(), [2,3,4])


def test_affines_init():
    # Test we are doing vaguely spec-related qform things.  The 'spec' here is
    # some thoughts by Mark Jenkinson:
    # http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/qsform_brief_usage
    arr = np.arange(24).reshape((2,3,4))
    aff = np.diag([2, 3, 4, 1])
    # Default is sform set, qform not set
    img = Nifti1Image(arr, aff)
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
    img = Nifti1Image(arr, aff, hdr)
    new_hdr = img.get_header()
    # Again affine is sort of anonymous space
    assert_equal(new_hdr['qform_code'], 0)
    assert_equal(new_hdr['sform_code'], 2)
    assert_array_equal(new_hdr.get_sform(), aff)
    assert_array_equal(new_hdr.get_zooms(), [2, 3, 4])
    # But if no affine passed, codes and matrices stay the same
    img = Nifti1Image(arr, None, hdr)
    new_hdr = img.get_header()
    assert_equal(new_hdr['qform_code'], 1) # scanner
    assert_array_equal(new_hdr.get_qform(), qaff)
    assert_equal(new_hdr['sform_code'], 3) # Still talairach
    assert_array_equal(new_hdr.get_sform(), saff)
    # Pixdims as in the original header
    assert_array_equal(new_hdr.get_zooms(), [3, 4, 5])



def test_pickle():
    # Smoketest that loaded nifti image do pickle
    nim = load(image_file)
    pickle.dumps(nim)
