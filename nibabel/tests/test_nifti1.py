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

from StringIO import StringIO

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

from ..testing import parametric, data_path

from . import test_analyze as tana
from .test_analyze import _log_chk

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
        hdr = self.header_class()
        for tests in tana.TestAnalyzeHeader.test_empty(self):
            yield tests
        yield assert_equal(hdr['magic'], 'ni1')
        yield assert_equal(hdr['scl_slope'], 1)
        yield assert_equal(hdr['vox_offset'], 0)

    def test_from_eg_file(self):
        hdr = Nifti1Header.from_fileobj(open(self.example_file))
        yield assert_equal(hdr.endianness, '<')
        yield assert_equal(hdr['magic'], 'ni1')
        yield assert_equal(hdr['sizeof_hdr'], 348)

    def test_nifti_log_checks(self):
        # in addition to analyze header checks
        HC = self.header_class
        # intercept and slope
        hdr = HC()
        hdr['scl_slope'] = 0
        fhdr, message, raiser = _log_chk(hdr, 30)
        yield assert_equal(fhdr['scl_slope'], 1)
        yield assert_equal(message, '"scl_slope" is 0.0; should !=0 '
                           'and be finite; setting "scl_slope" to 1')
        hdr = HC()
        hdr['scl_inter'] = np.nan # severity 30
        # NaN string representation can be odd on windows
        nan_str = '%s' % np.nan
        fhdr, message, raiser = _log_chk(hdr, 30)
        yield assert_equal(fhdr['scl_inter'], 0)
        yield assert_equal(message, '"scl_inter" is %s; should be '
                           'finite; setting "scl_inter" to 0' % nan_str)
        yield assert_raises(*raiser)
        # qfac
        hdr = HC()
        hdr['pixdim'][0] = 0
        fhdr, message, raiser = _log_chk(hdr, 20)
        yield assert_equal(fhdr['pixdim'][0], 1)
        yield assert_equal(message, 'pixdim[0] (qfac) should be 1 '
                           '(default) or -1; setting qfac to 1')
        # magic and offset
        hdr = HC()
        hdr['magic'] = 'ooh'
        fhdr, message, raiser = _log_chk(hdr, 45)
        yield assert_equal(fhdr['magic'], 'ooh')
        yield assert_equal(message, 'magic string "ooh" is not valid; '
                           'leaving as is, but future errors are likely')
        hdr['magic'] = 'n+1' # single file needs suitable offset
        hdr['vox_offset'] = 0
        fhdr, message, raiser = _log_chk(hdr, 40)
        yield assert_equal(fhdr['vox_offset'], 352)
        yield assert_equal(message, 'vox offset 0 too low for single '
                           'file nifti1; setting to minimum value '
                           'of 352')
        # qform, sform
        hdr = HC()
        hdr['qform_code'] = -1
        fhdr, message, raiser = _log_chk(hdr, 30)
        yield assert_equal(fhdr['qform_code'], 0)
        yield assert_equal(message, 'qform_code -1 not valid; '
                           'setting to 0')
        hdr = HC()
        hdr['sform_code'] = -1
        fhdr, message, raiser = _log_chk(hdr, 30)
        yield assert_equal(fhdr['sform_code'], 0)
        yield assert_equal(message, 'sform_code -1 not valid; '
                           'setting to 0')


class TestNifti1SingleHeader(TestNifti1PairHeader):

    header_class = Nifti1Header

    def test_empty(self):
        hdr = self.header_class()
        for tests in tana.TestAnalyzeHeader.test_empty(self):
            yield tests
        yield assert_equal(hdr['magic'], 'n+1')
        yield assert_equal(hdr['scl_slope'], 1)
        yield assert_equal(hdr['vox_offset'], 352)

    def test_binblock_is_file(self):
        # Override test that binary string is the same as the file on disk; in
        # the case of the single file version of the header, we need to append
        # the extension string (4 0s)
        hdr = self.header_class()
        str_io = StringIO()
        hdr.write_to(str_io)
        assert_equal(str_io.getvalue(), hdr.binaryblock + '\x00' * 4)


class TestNifti1Image(tana.TestAnalyzeImage):
    # class for testing images
    image_class = Nifti1Image


class TestNifti1Pair(tana.TestAnalyzeImage):
    image_class = Nifti1Pair


def test_datatypes():
    hdr = Nifti1Header()
    for code in data_type_codes.value_set():
        dt = data_type_codes.type[code]
        if dt == np.void:
            continue
        hdr.set_data_dtype(code)
        yield (assert_equal,
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
    yield assert_true, np.allclose(hdr.get_qform_quaternion(),
                       [1.0, 0, 0, 0])
    hdr['quatern_b'] = 1
    hdr['quatern_c'] = 0
    hdr['quatern_d'] = 0
    yield assert_true, np.allclose(hdr.get_qform_quaternion(),
                       [0, 1, 0, 0])


def test_qform():
    # Test roundtrip case
    ehdr = Nifti1Header()
    ehdr.set_qform(A)
    qA = ehdr.get_qform()
    yield assert_true, np.allclose(A, qA, atol=1e-5)
    yield assert_true, np.allclose(Z, ehdr['pixdim'][1:4])
    xfas = nifti1.xform_codes
    yield assert_true, ehdr['qform_code'] == xfas['scanner']
    ehdr.set_qform(A, 'aligned')
    yield assert_true, ehdr['qform_code'] == xfas['aligned']
    ehdr.set_qform(A, xfas['aligned'])
    yield assert_true, ehdr['qform_code'] == xfas['aligned']


def test_sform():
    # Test roundtrip case
    ehdr = Nifti1Header()
    ehdr.set_sform(A)
    sA = ehdr.get_sform()
    yield assert_true, np.allclose(A, sA, atol=1e-5)
    xfas = nifti1.xform_codes
    yield assert_true, ehdr['sform_code'] == xfas['scanner']
    ehdr.set_sform(A, 'aligned')
    yield assert_true, ehdr['sform_code'] == xfas['aligned']
    ehdr.set_sform(A, xfas['aligned'])
    yield assert_true, ehdr['sform_code'] == xfas['aligned']


@parametric
def test_dim_info():
    ehdr = Nifti1Header()
    yield assert_true(ehdr.get_dim_info() == (None, None, None))
    for info in ((0,2,1),
                 (None, None, None),
                 (0,2,None),
                 (0,None,None),
                 (None,2,1),
                 (None, None,1),
                 ):
        ehdr.set_dim_info(*info)
        yield assert_true(ehdr.get_dim_info() == info)


@parametric
def test_slice_times():
    hdr = Nifti1Header()
    # error if slice dimension not specified
    yield assert_raises(HeaderDataError, hdr.get_slice_times)
    hdr.set_dim_info(slice=2)
    # error if slice dimension outside shape
    yield assert_raises(HeaderDataError, hdr.get_slice_times)
    hdr.set_data_shape((1, 1, 7))
    # error if slice duration not set
    yield assert_raises(HeaderDataError, hdr.get_slice_times)    
    hdr.set_slice_duration(0.1)
    # We need a function to print out the Nones and floating point
    # values in a predictable way, for the tests below.
    _stringer = lambda val: val is not None and '%2.1f' % val or None
    _print_me = lambda s: map(_stringer, s)
    #The following examples are from the nifti1.h documentation.
    hdr['slice_code'] = slice_order_codes['sequential increasing']
    yield assert_equal(_print_me(hdr.get_slice_times()), 
                       ['0.0', '0.1', '0.2', '0.3', '0.4',
                        '0.5', '0.6'])
    hdr['slice_start'] = 1
    hdr['slice_end'] = 5
    yield assert_equal(_print_me(hdr.get_slice_times()),
        [None, '0.0', '0.1', '0.2', '0.3', '0.4', None])
    hdr['slice_code'] = slice_order_codes['sequential decreasing']
    yield assert_equal(_print_me(hdr.get_slice_times()),
        [None, '0.4', '0.3', '0.2', '0.1', '0.0', None])
    hdr['slice_code'] = slice_order_codes['alternating increasing']
    yield assert_equal(_print_me(hdr.get_slice_times()),
        [None, '0.0', '0.3', '0.1', '0.4', '0.2', None])
    hdr['slice_code'] = slice_order_codes['alternating decreasing']
    yield assert_equal(_print_me(hdr.get_slice_times()),
        [None, '0.2', '0.4', '0.1', '0.3', '0.0', None])
    hdr['slice_code'] = slice_order_codes['alternating increasing 2']
    yield assert_equal(_print_me(hdr.get_slice_times()),
        [None, '0.2', '0.0', '0.3', '0.1', '0.4', None])
    hdr['slice_code'] = slice_order_codes['alternating decreasing 2']
    yield assert_equal(_print_me(hdr.get_slice_times()),
        [None, '0.4', '0.1', '0.3', '0.0', '0.2', None])
    # test set
    hdr = Nifti1Header()
    hdr.set_dim_info(slice=2)
    # need slice dim to correspond with shape
    times = [None, 0.2, 0.4, 0.1, 0.3, 0.0, None]
    yield assert_raises(HeaderDataError, hdr.set_slice_times, times)
    hdr.set_data_shape([1, 1, 7])
    yield assert_raises(HeaderDataError,
                        hdr.set_slice_times,
                        times[:-1]) # wrong length
    yield assert_raises(HeaderDataError,
                        hdr.set_slice_times,
                        (None,) * len(times)) # all None
    n_mid_times = times[:]
    n_mid_times[3] = None
    yield assert_raises(HeaderDataError,
                        hdr.set_slice_times,
                        n_mid_times) # None in middle
    funny_times = times[:]
    funny_times[3] = 0.05
    yield assert_raises(HeaderDataError,
                        hdr.set_slice_times,
                        funny_times) # can't get single slice duration
    hdr.set_slice_times(times)
    yield assert_equal(hdr.get_value_label('slice_code'),
                       'alternating decreasing')
    yield assert_equal(hdr['slice_start'], 1)
    yield assert_equal(hdr['slice_end'], 5)
    yield assert_array_almost_equal(hdr['slice_duration'], 0.1)


@parametric
def test_intents():
    ehdr = Nifti1Header()
    ehdr.set_intent('t test', (10,), name='some score')
    yield assert_equal(ehdr.get_intent(),
                       ('t test', (10.0,), 'some score'))
    # invalid intent name
    yield assert_raises(KeyError,
                        ehdr.set_intent, 'no intention')
    # too many parameters
    yield assert_raises(HeaderDataError,
                        ehdr.set_intent, 
                        't test', (10,10))
    # too few parameters
    yield assert_raises(HeaderDataError,
                        ehdr.set_intent,
                        'f test', (10,))
    # check unset parameters are set to 0, and name to ''
    ehdr.set_intent('t test')
    yield assert_equal((ehdr['intent_p1'],
                        ehdr['intent_p2'],
                        ehdr['intent_p3']), (0,0,0))
    yield assert_equal(ehdr['intent_name'], '')
    ehdr.set_intent('t test', (10,))
    yield assert_equal((ehdr['intent_p2'],
                        ehdr['intent_p3']), (0,0))


@parametric
def test_set_slice_times():
    hdr = Nifti1Header()
    hdr.set_dim_info(slice=2)
    hdr.set_data_shape([1, 1, 7])
    hdr.set_slice_duration(0.1)
    times = [0] * 6
    yield assert_raises(HeaderDataError, hdr.set_slice_times, times)
    times = [None] * 7
    yield assert_raises(HeaderDataError, hdr.set_slice_times, times)
    times = [None, 0, 1, None, 3, 4, None]
    yield assert_raises(HeaderDataError, hdr.set_slice_times, times)
    times = [None, 0, 1, 2.1, 3, 4, None]
    yield assert_raises(HeaderDataError, hdr.set_slice_times, times)
    times = [None, 0, 4, 3, 2, 1, None]
    yield assert_raises(HeaderDataError, hdr.set_slice_times, times)
    times = [0, 1, 2, 3, 4, 5, 6]
    hdr.set_slice_times(times)
    yield assert_equal(hdr['slice_code'], 1)
    yield assert_equal(hdr['slice_start'], 0)
    yield assert_equal(hdr['slice_end'], 6)
    yield assert_equal(hdr['slice_duration'], 1.0)
    times = [None, 0, 1, 2, 3, 4, None]
    hdr.set_slice_times(times)
    yield assert_equal(hdr['slice_code'], 1)
    yield assert_equal(hdr['slice_start'], 1)
    yield assert_equal(hdr['slice_end'], 5)
    yield assert_equal(hdr['slice_duration'], 1.0)
    times = [None, 0.4, 0.3, 0.2, 0.1, 0, None]
    hdr.set_slice_times(times)
    yield assert_true(np.allclose(hdr['slice_duration'], 0.1))
    times = [None, 4, 3, 2, 1, 0, None]
    hdr.set_slice_times(times)
    yield assert_equal(hdr['slice_code'], 2)
    times = [None, 0, 3, 1, 4, 2, None]
    hdr.set_slice_times(times)
    yield assert_equal(hdr['slice_code'], 3)
    times = [None, 2, 4, 1, 3, 0, None]
    hdr.set_slice_times(times)
    yield assert_equal(hdr['slice_code'], 4)
    times = [None, 2, 0, 3, 1, 4, None]
    hdr.set_slice_times(times)
    yield assert_equal(hdr['slice_code'], 5)
    times = [None, 4, 1, 3, 0, 2, None]
    hdr.set_slice_times(times)
    yield assert_equal(hdr['slice_code'], 6)


def test_nifti1_images():
    shape = (2, 4, 6)
    npt = np.float32
    data = np.arange(np.prod(shape), dtype=npt).reshape(shape)
    affine = np.diag([1, 2, 3, 1])
    img = Nifti1Image(data, affine)
    assert_equal(img.get_shape(), shape)
    img.set_data_dtype(npt)
    stio = StringIO()
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


@parametric
def test_extension_basics():
    raw = '123'
    ext = Nifti1Extension('comment', raw)
    yield assert_true(ext.get_sizeondisk() == 16)
    yield assert_true(ext.get_content() == raw)
    yield assert_true(ext.get_code() == 6)


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
    assert_true(len(exts_container) == 2)
    assert_true(exts_container.count('comment') == 2)
    assert_true(exts_container.count('afni') == 0)
    assert_true(exts_container.get_codes() == [6, 6])
    assert_true((exts_container.get_sizeondisk()) % 16 == 0)
    # first extension should be short one
    assert_true(exts_container[0].get_content() == 'extcomment1')
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
    stio = StringIO()
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
    stio = StringIO()
    wnim.file_map['image'].fileobj = stio
    wnim.to_file_map()
    stio.seek(0)
    lnim = Nifti1Image.from_file_map(wnim.file_map)
    assert_equal(lnim.get_data_dtype(), np.int16)
    # the test below does not pass, because the slope and inter are
    # always reset from the data, by the image write
    raise SkipTest
    assert_equal(lnim.get_header().get_slope_inter(), (2, 8))


@parametric
def test_slope_inter():
    hdr = Nifti1Header()
    yield assert_equal(hdr.get_slope_inter(), (1.0, 0.0))
    hdr.set_slope_inter(2.2)
    yield assert_array_almost_equal(hdr.get_slope_inter(),
                                    (2.2, 0.0))
    hdr.set_slope_inter(None)
    yield assert_equal(hdr.get_slope_inter(),
                       (1.0, 0.0))
    hdr.set_slope_inter(2.2, 1.1)
    yield assert_array_almost_equal(hdr.get_slope_inter(),
                                    (2.2, 1.1))


@parametric
def test_xyzt_units():
    hdr = Nifti1Header()
    yield assert_equal(hdr.get_xyzt_units(), ('unknown', 'unknown'))
    hdr.set_xyzt_units('mm', 'sec')
    yield assert_equal(hdr.get_xyzt_units(), ('mm', 'sec'))
    hdr.set_xyzt_units()
    yield assert_equal(hdr.get_xyzt_units(), ('unknown', 'unknown'))


@parametric
def test_recoded_fields():
    hdr = Nifti1Header()
    yield assert_equal(hdr.get_value_label('qform_code'), 'unknown')
    hdr['qform_code'] = 3
    yield assert_equal(hdr.get_value_label('qform_code'), 'talairach')
    yield assert_equal(hdr.get_value_label('sform_code'), 'unknown')
    hdr['sform_code'] = 3
    yield assert_equal(hdr.get_value_label('sform_code'), 'talairach')
    yield assert_equal(hdr.get_value_label('intent_code'), 'none')
    hdr.set_intent('t test', (10,), name='some score')
    yield assert_equal(hdr.get_value_label('intent_code'), 't test')
    yield assert_equal(hdr.get_value_label('slice_code'), 'unknown')
    hdr['slice_code'] = 4 # alternating decreasing
    yield assert_equal(hdr.get_value_label('slice_code'),
                       'alternating decreasing')

