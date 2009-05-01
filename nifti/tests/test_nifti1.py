''' Tests for nifti reading package '''
import os
import tempfile

from StringIO import StringIO

import numpy as np

from numpy.testing import assert_array_equal
from nose.tools import assert_true, assert_equal, assert_raises

import nifti.testing as vit

from nifti.volumeutils import HeaderDataError
import nifti.nifti1 as nifti1
from nifti.nifti1 import Nifti1Header, Nifti1Image

from test_spm2analyze import TestSpm2AnalyzeHeader as _TSAH
from test_analyze import TestAnalyzeHeader

data_path, _ = os.path.split(__file__)
data_path = os.path.join(data_path, 'data')
header_file = os.path.join(data_path, 'nifti1.hdr')

# Example transformation matrix
R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]] # rotation matrix
Z = [2.0, 3.0, 4.0] # zooms
T = [20, 30, 40] # translations
A = np.eye(4)
A[:3,:3] = np.array(R) * Z # broadcasting does the job
A[:3,3] = T

class TestNiftiHeader(_TSAH):
    header_class = Nifti1Header
    example_file = header_file

    def test_empty(self):
        hdr = self.header_class()
        for tests in TestAnalyzeHeader.test_empty(self):
            yield tests
        yield vit.assert_equal, hdr['magic'], 'n+1'
        yield vit.assert_equal, hdr['scl_slope'], 1
        yield vit.assert_equal, hdr['vox_offset'], 352

    def test_from_eg_file(self):
        hdr = Nifti1Header.from_fileobj(open(self.example_file))
        yield vit.assert_equal, hdr.endianness, '<'
        yield vit.assert_equal, hdr['magic'], 'ni1'
        yield vit.assert_equal, hdr['sizeof_hdr'], 348


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
    
    
def test_dim_info():
    ehdr = Nifti1Header()
    ehdr.set_dim_info(0, 2, 1)
    yield assert_true, ehdr.get_dim_info() == (0, 2, 1)


def test_intents():
    ehdr = Nifti1Header()
    ehdr.set_intent('t test', (10,), name='some score')
    yield assert_equal, ehdr.get_intent(), ('t test', (10.0,), 'some score')
    yield (assert_raises, KeyError,
           ehdr.set_intent, 'no intention') # invalid intent name
    yield (assert_raises, HeaderDataError,
           ehdr.set_intent, 't test', (10,10)) # too many parameters
    yield (assert_raises, HeaderDataError,
           ehdr.set_intent, 'f test', (10,)) # too few parameters
    # check unset parameters are set to 0, and name to ''
    ehdr.set_intent('t test')
    yield assert_equal, (ehdr['intent_p1'],
                          ehdr['intent_p2'],
                          ehdr['intent_p3']), (0,0,0)
    yield assert_equal, ehdr['intent_name'], ''
    ehdr.set_intent('t test', (10,))
    yield assert_equal, (ehdr['intent_p2'], ehdr['intent_p3']), (0,0)


def test_set_slice_times():
    hdr = Nifti1Header()
    hdr.set_dim_info(slice=2)
    hdr.set_data_shape([1, 1, 7])
    hdr.set_slice_duration(0.1)
    times = [0] * 6
    yield assert_raises, HeaderDataError, hdr.set_slice_times, times 
    times = [None] * 7
    yield assert_raises, HeaderDataError, hdr.set_slice_times, times
    times = [None, 0, 1, None, 3, 4, None]
    yield assert_raises, HeaderDataError, hdr.set_slice_times, times
    times = [None, 0, 1, 2.1, 3, 4, None]
    yield assert_raises, HeaderDataError, hdr.set_slice_times, times
    times = [None, 0, 4, 3, 2, 1, None]
    yield assert_raises, HeaderDataError, hdr.set_slice_times, times
    times = [0, 1, 2, 3, 4, 5, 6]
    hdr.set_slice_times(times)
    yield assert_equal, hdr['slice_code'], 1
    yield assert_equal, hdr['slice_start'], 0
    yield assert_equal, hdr['slice_end'], 6
    yield assert_equal, hdr['slice_duration'], 1.0
    times = [None, 0, 1, 2, 3, 4, None]
    hdr.set_slice_times(times)
    yield assert_equal, hdr['slice_code'], 1
    yield assert_equal, hdr['slice_start'], 1
    yield assert_equal, hdr['slice_end'], 5
    yield assert_equal, hdr['slice_duration'], 1.0
    times = [None, 0.4, 0.3, 0.2, 0.1, 0, None]
    hdr.set_slice_times(times)
    yield assert_true, np.allclose(hdr['slice_duration'], 0.1)
    times = [None, 4, 3, 2, 1, 0, None]
    hdr.set_slice_times(times)
    yield assert_equal, hdr['slice_code'], 2
    times = [None, 0, 3, 1, 4, 2, None]
    hdr.set_slice_times(times)
    yield assert_equal, hdr['slice_code'], 3
    times = [None, 2, 4, 1, 3, 0, None]
    hdr.set_slice_times(times)
    yield assert_equal, hdr['slice_code'], 4
    times = [None, 2, 0, 3, 1, 4, None]
    hdr.set_slice_times(times)
    yield assert_equal, hdr['slice_code'], 5
    times = [None, 4, 1, 3, 0, 2, None]
    hdr.set_slice_times(times)
    yield assert_equal, hdr['slice_code'], 6


def test_nifti1_images():
    shape = (2, 4, 6)
    npt = np.float32
    data = np.arange(np.prod(shape), dtype=npt).reshape(shape)
    affine = np.diag([1, 2, 3, 1])
    img = Nifti1Image(data, affine)
    yield assert_equal, img.get_shape(), shape
    img.set_data_dtype(npt)
    stio = StringIO()
    files = {'header': stio, 'image': stio}
    img.to_files(files)
    stio.seek(0)
    img2 = Nifti1Image.from_files(files)
    yield assert_array_equal, img2.get_data(), data
    '''
    try:
        _, fname = tempfile.mkstemp('.nii.gz')
        img.to_filespec(fname)
        img3 = nifti1.load(fname)
        yield assert_true, isinstance(img3, img.__class__)
        yield assert_array_equal, img3.get_data(), data
        yield assert_equal, img3.get_header(), img.get_header()
    finally:
        os.unlink(fname)
    '''
