''' Tests for nifti reading package '''
import os
import tempfile

from StringIO import StringIO

import numpy as np

from nibabel.spatialimages import HeaderDataError
import nibabel.nifti1 as nifti1
from nibabel.nifti1 import load, Nifti1Header, Nifti1Image, \
    Nifti1Extension, data_type_codes, extension_codes, \
    slice_order_codes

from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_true, assert_false, assert_equal, \
    assert_raises

from nibabel.testing import parametric, data_path

import test_analyze as tana

header_file = os.path.join(data_path, 'nifti1.hdr')
image_file = os.path.join(data_path, 'example4d.nii.gz')


# Example transformation matrix
R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]] # rotation matrix
Z = [2.0, 3.0, 4.0] # zooms
T = [20, 30, 40] # translations
A = np.eye(4)
A[:3,:3] = np.array(R) * Z # broadcasting does the job
A[:3,3] = T


class TestNiftiHeader(tana.TestAnalyzeHeader):
    header_class = Nifti1Header
    example_file = header_file

    def test_empty(self):
        hdr = self.header_class()
        for tests in tana.TestAnalyzeHeader.test_empty(self):
            yield tests
        yield assert_equal(hdr['magic'], 'n+1')
        yield assert_equal(hdr['scl_slope'], 1)
        yield assert_equal(hdr['vox_offset'], 352)

    def test_from_eg_file(self):
        hdr = Nifti1Header.from_fileobj(open(self.example_file))
        yield assert_equal(hdr.endianness, '<')
        yield assert_equal(hdr['magic'], 'ni1')
        yield assert_equal(hdr['sizeof_hdr'], 348)


class TestNifti1Image(tana.AnalyzeImage):
    # class for testing images
    image_class = Nifti1Image
    header_class = Nifti1Header


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


'''
def test_checked():
    HC = Nifti1Header
    def check_cf(hdr, log_level, error_level, fixed):
        # return tests for check_fix
        hdr = HC.from_header(hdr, 
    ehf = vn.empty_header
    ckdf = vn.checked
    sso = sys.stdout
    # test header gives no warnings at any severity
    log = StringIO()
    hdr2 = ckdf(hdr, log=log, severity=0.0)
    yield assert_equal, log.tell(), 0
    # our headers may break it though
    eh = ehf()
    eh['sizeof_hdr'] = 350 # severity 2.0
    ehc = ckdf(eh, sso, 2.1)
    yield assert_equal, ehc.sizeof_hdr, 348
    yield assert_raises, HeaderDataError, ckdf, eh, None, 2.0
    eh = ehf()
    eh.pixdim[0] = 0 # severity 1.0
    ehc = ckdf(eh, sso, 1.1)
    yield assert_equal, ehc.pixdim[0], 1
    yield assert_raises, HeaderDataError, ckdf, eh, None, 1.0
    eh = ehf()
    eh.pixdim[1] = -1 # severity 4.0
    ehc = ckdf(eh, sso, 4.1)
    yield assert_equal, ehc.pixdim[1], 1
    yield assert_raises, HeaderDataError, ckdf, eh, None, 4.0
    eh = ehf()
    eh.datatype = -1 # severity 9.0
    ehc = ckdf(eh, sso, 9.1)
    yield assert_equal, ehc.datatype, -1 # left as is
    yield assert_raises, HeaderDataError, ckdf, eh, None, 9.0
    eh = ehf()
    eh.datatype = 2
    eh.bitpix = 16 # severity 1.0
    ehc = ckdf(eh, sso, 1.1)
    yield assert_equal, ehc.bitpix, 8 
    yield assert_raises, HeaderDataError, ckdf, eh, None, 1.0
    eh = ehf()
    eh.magic = 'ni1'
    eh.vox_offset = 1 # severity 8.0
    ehc = ckdf(eh, sso, 8.1)
    yield assert_equal, ehc.vox_offset, 1
    yield assert_raises, HeaderDataError, ckdf, eh, None, 8.0
    eh.magic = 'n+1'
    # vox offset now wrong for single file - severity 9.0
    ehc = ckdf(eh, sso, 9.1)
    yield assert_equal, ehc.vox_offset, 352
    yield assert_raises, HeaderDataError, ckdf, eh, None, 9.0
    eh.vox_offset = 353
    # now theoretically valid but not for SPM -> 3.0
    ehc = ckdf(eh, sso, 3.1)
    yield assert_equal, ehc.vox_offset, 353
    yield assert_raises, HeaderDataError, ckdf, eh, None, 3.0
    # bad magic - severity 9.5
    eh = ehf()
    eh.magic = 'nul'
    ehc = ckdf(eh, sso, 9.6)
    yield assert_equal, ehc.magic, 'nul' # leave
    yield assert_raises, HeaderDataError, ckdf, eh, None, 9.5
    # qform, sform transforms, severity 3.0
    eh = ehf()
    eh.qform_code = -1
    ehc = ckdf(eh, sso, 3.1)
    yield assert_equal, ehc.qform_code, 0
    yield assert_raises, HeaderDataError, ckdf, eh, None, 3.0
    eh = ehf()
    eh.sform_code = -1
    ehc = ckdf(eh, sso, 3.1)
    yield assert_equal, ehc.sform_code, 0
    yield assert_raises, HeaderDataError, ckdf, eh, None, 3.0
'''

    
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
    yield assert_equal(hdr.get_field_label('slice_code'),
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


@parametric
def test_nifti1_images():
    shape = (2, 4, 6)
    npt = np.float32
    data = np.arange(np.prod(shape), dtype=npt).reshape(shape)
    affine = np.diag([1, 2, 3, 1])
    img = Nifti1Image(data, affine)
    yield assert_equal(img.get_shape(), shape)
    img.set_data_dtype(npt)
    stio = StringIO()
    img.file_map['image'].fileobj = stio
    img.to_file_map()
    img2 = Nifti1Image.from_file_map(img.file_map)
    yield assert_array_equal(img2.get_data(), data)
    for ext in ('.gz', '.bz2'):
        try:
            _, fname = tempfile.mkstemp('.nii' + ext)
            img.to_filename(fname)
            img3 = Nifti1Image.load(fname)
            yield assert_true(isinstance(img3, img.__class__))
            yield assert_array_equal(img3.get_data(), data)
            yield assert_equal(img3.get_header(), img.get_header())
        finally:
            os.unlink(fname)


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


@parametric
def test_nifti_extensions():
    nim = load(image_file)
    # basic checks of the available extensions
    ext = nim.extra['extensions']
    yield assert_true(len(ext) == 2)
    yield assert_true(ext.count('comment') == 2)
    yield assert_true(ext.count('afni') == 0)
    yield assert_true(ext.get_codes() == [6, 6])
    yield assert_true((ext.get_sizeondisk() - 4) % 16 == 0)
    # first extension should be short one
    yield assert_true(ext[0].get_content() == 'extcomment1')
    # add one
    afniext = Nifti1Extension('afni', '<xml></xml>')
    ext.append(afniext)
    yield assert_true(ext.get_codes() == [6, 6, 4])
    yield assert_true(ext.count('comment') == 2)
    yield assert_true(ext.count('afni') == 1)
    yield assert_true((ext.get_sizeondisk() - 4) % 16 == 0)
    # delete one
    del ext[1]
    yield assert_true(ext.get_codes() == [6, 4])
    yield assert_true(ext.count('comment') == 1)
    yield assert_true(ext.count('afni') == 1)


@parametric
def test_loadsave_cycle():
    nim = load(image_file)
    # ensure we have extensions
    yield assert_true(nim.extra.has_key('extensions'))
    yield assert_true(len(nim.extra['extensions']))
    # write into the air ;-)
    stio = StringIO()
    nim.file_map['image'].fileobj = stio
    nim.to_file_map()
    stio.seek(0)
    # reload
    lnim = Nifti1Image.from_file_map(nim.file_map)
    yield assert_true(lnim.extra.has_key('extensions'))
    yield assert_equal(nim.extra['extensions'],
                       lnim.extra['extensions'])


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
def test_recoded_fields():
    hdr = Nifti1Header()
    yield assert_equal(hdr.get_field_label('qform_code'), 'unknown')
    hdr['qform_code'] = 3
    yield assert_equal(hdr.get_field_label('qform_code'), 'talairach')
    yield assert_equal(hdr.get_field_label('sform_code'), 'unknown')
    hdr['sform_code'] = 3
    yield assert_equal(hdr.get_field_label('sform_code'), 'talairach')
    yield assert_equal(hdr.get_field_label('intent_code'), 'none')
    hdr.set_intent('t test', (10,), name='some score')
    yield assert_equal(hdr.get_field_label('intent_code'), 't test')
    yield assert_equal(hdr.get_field_label('slice_code'), 'unknown')
    hdr['slice_code'] = 4 # alternating decreasing
    yield assert_equal(hdr.get_field_label('slice_code'),
                       'alternating decreasing')

