""" Testing DICOM wrappers
"""

from os.path import join as pjoin, dirname
import gzip

import numpy as np

try:
    import dicom
except ImportError:
    have_dicom = False
else:
    have_dicom = True
dicom_test = np.testing.dec.skipif(not have_dicom,
                                   'could not import pydicom')

from .. import dicomwrappers as didw
from .. import dicomreaders as didr

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

IO_DATA_PATH = pjoin(dirname(__file__), 'data')
DATA_FILE = pjoin(IO_DATA_PATH, 'siemens_dwi_1000.dcm.gz')
if have_dicom:
    DATA = dicom.read_file(gzip.open(DATA_FILE))
else:
    DATA = None
DATA_FILE_B0 = pjoin(IO_DATA_PATH, 'siemens_dwi_0.dcm.gz')
DATA_FILE_SLC_NORM = pjoin(IO_DATA_PATH, 'csa_slice_norm.dcm')

# This affine from our converted image was shown to match our image
# spatially with an image from SPM DICOM conversion. We checked the
# matching with SPM check reg.  We have flipped the first and second
# rows to allow for rows, cols tranpose in current return compared to
# original case.
EXPECTED_AFFINE = np.array(
    [[ -1.796875, 0, 0, 115],
     [0, -1.79684984, -0.01570896, 135.028779],
     [0, -0.00940843750, 2.99995887, -78.710481],
     [0, 0, 0, 1]])[:,[1,0,2,3]]

# from Guys and Matthew's SPM code, undoing SPM's Y flip, and swapping
# first two values in vector, to account for data rows, cols difference.
EXPECTED_PARAMS = [992.05050247, (0.00507649,
                                  0.99997450,
                                  -0.005023611)]

@dicom_test
def test_wrappers():
    # test direct wrapper calls
    # first with empty data
    for maker, kwargs in ((didw.Wrapper,{}),
                          (didw.SiemensWrapper, {}),
                          (didw.MosaicWrapper, {'n_mosaic':10})):
        dw = maker(**kwargs)
        assert_equal(dw.get('InstanceNumber'), None)
        assert_equal(dw.get('AcquisitionNumber'), None)
        assert_raises(KeyError, dw.__getitem__, 'not an item')
        assert_raises(didw.WrapperError, dw.get_data)
        assert_raises(didw.WrapperError, dw.get_affine)
    for klass in (didw.Wrapper, didw.SiemensWrapper):
        dw = klass()
        assert_false(dw.is_mosaic)
    for maker in (didw.wrapper_from_data,
                  didw.Wrapper,
                  didw.SiemensWrapper,
                  didw.MosaicWrapper
                  ):
        dw = maker(DATA)
        assert_equal(dw.get('InstanceNumber'), 2)
        assert_equal(dw.get('AcquisitionNumber'), 2)
        assert_raises(KeyError, dw.__getitem__, 'not an item')
    for maker in (didw.MosaicWrapper, didw.wrapper_from_data):
        assert_true(dw.is_mosaic)


@dicom_test
def test_wrapper_from_data():
    # test wrapper from data, wrapper from file
    for dw in (didw.wrapper_from_data(DATA),
               didw.wrapper_from_file(DATA_FILE)):
        assert_equal(dw.get('InstanceNumber'), 2)
        assert_equal(dw.get('AcquisitionNumber'), 2)
        assert_raises(KeyError, dw.__getitem__, 'not an item')
        assert_true(dw.is_mosaic)
        assert_array_almost_equal(
            np.dot(didr.DPCS_TO_TAL, dw.get_affine()),
            EXPECTED_AFFINE)


@dicom_test
def test_dwi_params():
    dw = didw.wrapper_from_data(DATA)
    b_matrix = dw.b_matrix
    assert_equal(b_matrix.shape, (3,3))
    q = dw.q_vector
    b = np.sqrt(np.sum(q * q)) # vector norm
    g = q / b
    assert_array_almost_equal(b, EXPECTED_PARAMS[0])
    assert_array_almost_equal(g, EXPECTED_PARAMS[1])


@dicom_test
def test_vol_matching():
    # make the Siemens wrapper, check it compares True against itself
    dw_siemens = didw.wrapper_from_data(DATA)
    assert_true(dw_siemens.is_mosaic)
    assert_true(dw_siemens.is_csa)
    assert_true(dw_siemens.is_same_series(dw_siemens))
    # make plain wrapper, compare against itself
    dw_plain = didw.Wrapper(DATA)
    assert_false(dw_plain.is_mosaic)
    assert_false(dw_plain.is_csa)
    assert_true(dw_plain.is_same_series(dw_plain))
    # specific vs plain wrapper compares False, because the Siemens
    # wrapper has more non-empty information
    assert_false(dw_plain.is_same_series(dw_siemens))
    # and this should be symmetric
    assert_false(dw_siemens.is_same_series(dw_plain))
    # we can even make an empty wrapper.  This compares True against
    # itself but False against the others
    dw_empty = didw.Wrapper()
    assert_true(dw_empty.is_same_series(dw_empty))
    assert_false(dw_empty.is_same_series(dw_plain))
    assert_false(dw_plain.is_same_series(dw_empty))
    # Just to check the interface, make a pretend signature-providing
    # object.
    class C(object):
        series_signature = {}
    assert_true(dw_empty.is_same_series(C()))


@dicom_test
def test_slice_indicator():
    dw_0 = didw.wrapper_from_file(DATA_FILE_B0)
    dw_1000 = didw.wrapper_from_data(DATA)
    z = dw_0.slice_indicator
    assert_false(z is None)
    assert_equal(z, dw_1000.slice_indicator)
    dw_empty = didw.Wrapper()
    assert_true(dw_empty.slice_indicator is None)


@dicom_test
def test_orthogonal():
    #Test that the slice normal is sufficiently orthogonal
    dw = didw.wrapper_from_file(DATA_FILE_SLC_NORM)
    R = dw.rotation_matrix
    assert  np.allclose(np.eye(3),
                        np.dot(R, R.T),
                        atol=1e-6)
                    
@dicom_test
def test_use_csa_sign():
    #Test that we get the same slice normal, even after swapping the iop 
    #directions
    dw = didw.wrapper_from_file(DATA_FILE_SLC_NORM)
    iop = dw.image_orient_patient
    dw.image_orient_patient = np.c_[iop[:,1], iop[:,0]]
    dw2 = didw.wrapper_from_file(DATA_FILE_SLC_NORM)
    assert np.allclose(dw.slice_normal, dw2.slice_normal)

@dicom_test
def test_assert_parallel():
    #Test that we get an AssertionError if the cross product and the CSA 
    #slice normal are not parallel
    dw = didw.wrapper_from_file(DATA_FILE_SLC_NORM)
    dw.image_orient_patient = np.c_[[1., 0., 0.], [0., 1., 0.]]
    assert_raises(AssertionError, dw.__getattribute__, 'slice_normal')
