""" Testing DICOM wrappers
"""

from os.path import join as pjoin, dirname
import gzip
from hashlib import sha1
from decimal import Decimal
from copy import copy

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

from unittest import TestCase
from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_not_equal, assert_raises)

from numpy.testing import assert_array_equal, assert_array_almost_equal

IO_DATA_PATH = pjoin(dirname(__file__), 'data')
DATA_FILE = pjoin(IO_DATA_PATH, 'siemens_dwi_1000.dcm.gz')
DATA_FILE_PHILIPS = pjoin(IO_DATA_PATH, 'philips_mprage.dcm.gz')
if have_dicom:
    DATA = dicom.read_file(gzip.open(DATA_FILE))
    DATA_PHILIPS = dicom.read_file(gzip.open(DATA_FILE_PHILIPS))
else:
    DATA = None
    DATA_PHILIPS = None
DATA_FILE_B0 = pjoin(IO_DATA_PATH, 'siemens_dwi_0.dcm.gz')
DATA_FILE_SLC_NORM = pjoin(IO_DATA_PATH, 'csa_slice_norm.dcm')
DATA_FILE_DEC_RSCL = pjoin(IO_DATA_PATH, 'decimal_rescale.dcm')
DATA_FILE_4D = pjoin(IO_DATA_PATH, '4d_multiframe_test.dcm')

# This affine from our converted image was shown to match our image spatially
# with an image from SPM DICOM conversion. We checked the matching with SPM
# check reg.  We have flipped the first and second rows to allow for rows, cols
# transpose in current return compared to original case.
EXPECTED_AFFINE = np.array(  # do this for philips?
    [[-1.796875, 0, 0, 115],
     [0, -1.79684984, -0.01570896, 135.028779],
     [0, -0.00940843750, 2.99995887, -78.710481],
     [0, 0, 0, 1]])[:, [1, 0, 2, 3]]

# from Guys and Matthew's SPM code, undoing SPM's Y flip, and swapping first two
# values in vector, to account for data rows, cols difference.
EXPECTED_PARAMS = [992.05050247, (0.00507649,
                                  0.99997450,
                                  -0.005023611)]

@dicom_test
def test_wrappers():
    # test direct wrapper calls
    # first with empty or minimal data
    multi_minimal = {
        'PerFrameFunctionalGroupsSequence': [None],
        'SharedFunctionalGroupsSequence': [None]}
    for maker, args in ((didw.Wrapper, ({},)),
                        (didw.SiemensWrapper, ({},)),
                        (didw.MosaicWrapper, ({}, None, 10)),
                        (didw.MultiframeWrapper, (multi_minimal,))):
        dw = maker(*args)
        assert_equal(dw.get('InstanceNumber'), None)
        assert_equal(dw.get('AcquisitionNumber'), None)
        assert_raises(KeyError, dw.__getitem__, 'not an item')
        assert_raises(didw.WrapperError, dw.get_data)
        assert_raises(didw.WrapperError, dw.get_affine)
        assert_raises(TypeError, maker)
        # Check default attributes
        if not maker is didw.MosaicWrapper:
            assert_false(dw.is_mosaic)
        assert_equal(dw.b_matrix, None)
        assert_equal(dw.q_vector, None)
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
        dw = maker(DATA)
        assert_true(dw.is_mosaic)
    # DATA is not a Multiframe DICOM file
    assert_raises(didw.WrapperError, didw.MultiframeWrapper, DATA)


def test_get_from_wrapper():
    # Test that 'get', and __getitem__ work as expected for underlying dicom
    # data
    dcm_data = {'some_key': 'some value'}
    dw = didw.Wrapper(dcm_data)
    assert_equal(dw.get('some_key'), 'some value')
    assert_equal(dw.get('some_other_key'), None)
    # Getitem uses the same dictionary access
    assert_equal(dw['some_key'], 'some value')
    # And raises a WrapperError for missing keys
    assert_raises(KeyError, dw.__getitem__, 'some_other_key')
    # Test we don't use attributes for get

    class FakeData(dict):
        pass
    d = FakeData()
    d.some_key = 'another bit of data'
    dw = didw.Wrapper(d)
    assert_equal(dw.get('some_key'), None)
    # Check get defers to dcm_data get

    class FakeData2(object):
        def get(self, key, default):
            return 1
    d = FakeData2()
    d.some_key = 'another bit of data'
    dw = didw.Wrapper(d)
    assert_equal(dw.get('some_key'), 1)


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
    for dw in (didw.wrapper_from_data(DATA_PHILIPS),
               didw.wrapper_from_file(DATA_FILE_PHILIPS)):
        assert_equal(dw.get('InstanceNumber'), 1)
        assert_equal(dw.get('AcquisitionNumber'), 3)
        assert_raises(KeyError, dw.__getitem__, 'not an item')
        assert_true(dw.is_multiframe)
    # Another CSA file
    dw = didw.wrapper_from_file(DATA_FILE_SLC_NORM)
    assert_true(dw.is_mosaic)
    # Check that multiframe requires minimal set of DICOM tags
    fake_data = dict()
    fake_data['SOPClassUID'] = '1.2.840.10008.5.1.4.1.1.4.2'
    dw = didw.wrapper_from_data(fake_data)
    assert_false(dw.is_multiframe)
    # use the correct SOPClassUID
    fake_data['SOPClassUID'] = '1.2.840.10008.5.1.4.1.1.4.1'
    assert_raises(didw.WrapperError, didw.wrapper_from_data, fake_data)
    fake_data['PerFrameFunctionalGroupsSequence'] = [None]
    assert_raises(didw.WrapperError, didw.wrapper_from_data, fake_data)
    fake_data['SharedFunctionalGroupsSequence'] = [None]
    # minimal set should now be met
    dw = didw.wrapper_from_data(fake_data)
    assert_true(dw.is_multiframe)


@dicom_test
def test_wrapper_args_kwds():
    # Test we can pass args, kwargs to dicom.read_file
    dcm = didw.wrapper_from_file(DATA_FILE)
    data = dcm.get_data()
    # Passing in non-default arg for defer_size
    dcm2 = didw.wrapper_from_file(DATA_FILE, np.inf)
    assert_array_equal(data, dcm2.get_data())
    # Passing in non-default arg for defer_size with kwds
    dcm2 = didw.wrapper_from_file(DATA_FILE, defer_size=np.inf)
    assert_array_equal(data, dcm2.get_data())
    # Trying to read non-dicom file raises pydicom error, usually
    csa_fname = pjoin(IO_DATA_PATH, 'csa2_b0.bin')
    assert_raises(dicom.filereader.InvalidDicomError,
                  didw.wrapper_from_file,
                  csa_fname)
    # We can force the read, in which case rubbish returns
    dcm_malo = didw.wrapper_from_file(csa_fname, force=True)
    assert_false(dcm_malo.is_mosaic)


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
def test_q_vector_etc():
    # Test diffusion params in wrapper classes
    # Default is no q_vector, b_value, b_vector
    dw = didw.Wrapper(DATA)
    assert_equal(dw.q_vector, None)
    assert_equal(dw.b_value, None)
    assert_equal(dw.b_vector, None)
    for pos in range(3):
        q_vec = np.zeros((3,))
        q_vec[pos] = 10.
        # Reset wrapped dicom to refresh one_time property
        dw = didw.Wrapper(DATA)
        dw.q_vector = q_vec
        assert_array_equal(dw.q_vector, q_vec)
        assert_equal(dw.b_value, 10)
        assert_array_equal(dw.b_vector, q_vec / 10.)
    # Reset wrapped dicom to refresh one_time property
    dw = didw.Wrapper(DATA)
    dw.q_vector = np.array([0, 0, 1e-6])
    assert_equal(dw.b_value, 0)
    assert_array_equal(dw.b_vector, np.zeros((3,)))
    # Test MosaicWrapper
    sdw = didw.MosaicWrapper(DATA)
    exp_b, exp_g = EXPECTED_PARAMS
    assert_array_almost_equal(sdw.q_vector, exp_b * np.array(exp_g), 5)
    assert_array_almost_equal(sdw.b_value, exp_b)
    assert_array_almost_equal(sdw.b_vector, exp_g)
    # Reset wrapped dicom to refresh one_time property
    sdw = didw.MosaicWrapper(DATA)
    sdw.q_vector = np.array([0, 0, 1e-6])
    assert_equal(sdw.b_value, 0)
    assert_array_equal(sdw.b_vector, np.zeros((3,)))


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
    dw_empty = didw.Wrapper({})
    assert_true(dw_empty.is_same_series(dw_empty))
    assert_false(dw_empty.is_same_series(dw_plain))
    assert_false(dw_plain.is_same_series(dw_empty))
    # Just to check the interface, make a pretend signature-providing
    # object.

    class C(object):
        series_signature = {}
    assert_true(dw_empty.is_same_series(C()))

    # make the Philips wrapper, check it compares True against itself
    dw_philips = didw.wrapper_from_data(DATA_PHILIPS)
    assert_true(dw_philips.is_multiframe)
    assert_true(dw_philips.is_same_series(dw_philips))
    # make plain wrapper, compare against itself
    dw_plain_philips = didw.Wrapper(DATA)
    assert_false(dw_plain_philips.is_multiframe)
    assert_true(dw_plain_philips.is_same_series(dw_plain_philips))
    # specific vs plain wrapper compares False, because the Philips
    # wrapper has more non-empty information
    assert_false(dw_plain_philips.is_same_series(dw_philips))
    # and this should be symmetric
    assert_false(dw_philips.is_same_series(dw_plain_philips))
    # we can even make an empty wrapper.  This compares True against
    # itself but False against the others
    dw_empty = didw.Wrapper({})
    assert_true(dw_empty.is_same_series(dw_empty))
    assert_false(dw_empty.is_same_series(dw_plain_philips))
    assert_false(dw_plain_philips.is_same_series(dw_empty))


@dicom_test
def test_slice_indicator():
    dw_0 = didw.wrapper_from_file(DATA_FILE_B0)
    dw_1000 = didw.wrapper_from_data(DATA)
    z = dw_0.slice_indicator
    assert_false(z is None)
    assert_equal(z, dw_1000.slice_indicator)
    dw_empty = didw.Wrapper({})
    assert_true(dw_empty.slice_indicator is None)


@dicom_test
def test_orthogonal():
    # Test that the slice normal is sufficiently orthogonal
    dw = didw.wrapper_from_file(DATA_FILE_SLC_NORM)
    R = dw.rotation_matrix
    assert_true(np.allclose(np.eye(3), np.dot(R, R.T), atol=1e-6))

    # Test the threshold for rotation matrix orthogonality
    d = {}
    d['ImageOrientationPatient'] = [0, 1, 0, 1, 0, 0]
    dw = didw.wrapper_from_data(d)
    assert_array_equal(dw.rotation_matrix, np.eye(3))
    d['ImageOrientationPatient'] = [1e-5, 1, 0, 1, 0, 0]
    dw = didw.wrapper_from_data(d)
    assert_array_almost_equal(dw.rotation_matrix, np.eye(3), 5)
    d['ImageOrientationPatient'] = [1e-4, 1, 0, 1, 0, 0]
    dw = didw.wrapper_from_data(d)
    assert_raises(didw.WrapperPrecisionError, getattr, dw, 'rotation_matrix')


@dicom_test
def test_rotation_matrix():
    # Test rotation matrix and slice normal
    d = {}
    d['ImageOrientationPatient'] = [0, 1, 0, 1, 0, 0]
    dw = didw.wrapper_from_data(d)
    assert_array_equal(dw.rotation_matrix, np.eye(3))
    d['ImageOrientationPatient'] = [1, 0, 0, 0, 1, 0]
    dw = didw.wrapper_from_data(d)
    assert_array_equal(dw.rotation_matrix, [[0, 1, 0],
                                            [1, 0, 0],
                                            [0, 0, -1]])


@dicom_test
def test_use_csa_sign():
    #Test that we get the same slice normal, even after swapping the iop 
    #directions
    dw = didw.wrapper_from_file(DATA_FILE_SLC_NORM)
    iop = dw.image_orient_patient
    dw.image_orient_patient = np.c_[iop[:,1], iop[:,0]]
    dw2 = didw.wrapper_from_file(DATA_FILE_SLC_NORM)
    assert_true(np.allclose(dw.slice_normal, dw2.slice_normal))


@dicom_test
def test_assert_parallel():
    #Test that we get an AssertionError if the cross product and the CSA 
    #slice normal are not parallel
    dw = didw.wrapper_from_file(DATA_FILE_SLC_NORM)
    dw.image_orient_patient = np.c_[[1., 0., 0.], [0., 1., 0.]]
    assert_raises(AssertionError, dw.__getattribute__, 'slice_normal')


@dicom_test
def test_decimal_rescale():
    #Test that we don't get back a data array with dtype np.object when our
    #rescale slope is a decimal
    dw = didw.wrapper_from_file(DATA_FILE_DEC_RSCL)
    assert_not_equal(dw.get_data().dtype, np.object)


def fake_frames(seq_name, field_name, value_seq):
    """ Make fake frames for multiframe testing

    Parameters
    ----------
    seq_name : str
        name of sequence
    field_name : str
        name of field within sequence
    value_seq : length N sequence
        sequence of values

    Returns
    -------
    frame_seq : length N list
        each element in list is obj.<seq_name>[0].<field_name> =
        value_seq[n] for n in range(N)
    """
    class Fake(object): pass
    frames = []
    for value in value_seq:
        fake_frame = Fake()
        fake_element = Fake()
        setattr(fake_element, field_name, value)
        setattr(fake_frame, seq_name, [fake_element])
        frames.append(fake_frame)
    return frames


class TestMultiFrameWrapper(TestCase):
    # Test MultiframeWrapper
    MINIMAL_MF = {
        # Minimal contents of dcm_data for this wrapper
         'PerFrameFunctionalGroupsSequence': [None],
         'SharedFunctionalGroupsSequence': [None]}
    WRAPCLASS = didw.MultiframeWrapper

    def test_shape(self):
        # Check the shape algorithm
        fake_mf = copy(self.MINIMAL_MF)
        MFW = self.WRAPCLASS
        dw = MFW(fake_mf)
        # No rows, cols, raise WrapperError
        assert_raises(didw.WrapperError, getattr, dw, 'image_shape')
        fake_mf['Rows'] = 64
        assert_raises(didw.WrapperError, getattr, dw, 'image_shape')
        fake_mf.pop('Rows')
        fake_mf['Columns'] = 64
        assert_raises(didw.WrapperError, getattr, dw, 'image_shape')
        fake_mf['Rows'] = 32
        # Missing frame data, raise AssertionError
        assert_raises(AssertionError, getattr, dw, 'image_shape')
        fake_mf['NumberOfFrames'] = 4
        # PerFrameFunctionalGroupsSequence does not match NumberOfFrames
        assert_raises(AssertionError, getattr, dw, 'image_shape')
        # Make some fake frame data for 3D
        def my_fake_frames(div_seq):
            return fake_frames('FrameContentSequence',
                               'DimensionIndexValues',
                               div_seq)
        div_seq = ((1, 1), (1, 2), (1, 3), (1, 4))
        frames = my_fake_frames(div_seq)
        fake_mf['PerFrameFunctionalGroupsSequence'] = frames
        assert_equal(MFW(fake_mf).image_shape, (32, 64, 4))
        # Check stack number matching
        div_seq = ((1, 1), (1, 2), (1, 3), (2, 4))
        frames = my_fake_frames(div_seq)
        fake_mf['PerFrameFunctionalGroupsSequence'] = frames
        assert_raises(didw.WrapperError, getattr, MFW(fake_mf), 'image_shape')
        # Make some fake frame data for 4D
        fake_mf['NumberOfFrames'] = 6
        div_seq = ((1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 2, 2),
                (1, 1, 3), (1, 2, 3))
        frames = my_fake_frames(div_seq)
        fake_mf['PerFrameFunctionalGroupsSequence'] = frames
        assert_equal(MFW(fake_mf).image_shape, (32, 64, 2, 3))
        # Check stack number matching for 4D
        div_seq = ((1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 2, 2),
                (1, 1, 3), (2, 2, 3))
        frames = my_fake_frames(div_seq)
        fake_mf['PerFrameFunctionalGroupsSequence'] = frames
        assert_raises(didw.WrapperError, getattr, MFW(fake_mf), 'image_shape')
        # Check indices can be non-contiguous
        div_seq = ((1, 1, 1), (1, 2, 1), (1, 1, 3), (1, 2, 3))
        frames = my_fake_frames(div_seq)
        fake_mf['NumberOfFrames'] = 4
        fake_mf['PerFrameFunctionalGroupsSequence'] = frames
        assert_equal(MFW(fake_mf).image_shape, (32, 64, 2, 2))
        # Check indices can include zero
        div_seq = ((1, 1, 0), (1, 2, 0), (1, 1, 3), (1, 2, 3))
        frames = my_fake_frames(div_seq)
        fake_mf['NumberOfFrames'] = 4
        fake_mf['PerFrameFunctionalGroupsSequence'] = frames
        assert_equal(MFW(fake_mf).image_shape, (32, 64, 2, 2))

    def test_iop(self):
        # Test Image orient patient for multiframe
        fake_mf = copy(self.MINIMAL_MF)
        MFW = self.WRAPCLASS
        dw = MFW(fake_mf)
        assert_raises(didw.WrapperError, getattr, dw, 'image_orient_patient')
        # Make a fake frame
        fake_frame = fake_frames('PlaneOrientationSequence',
                                 'ImageOrientationPatient',
                                 [[0, 1, 0, 1, 0, 0]])[0]
        fake_mf['SharedFunctionalGroupsSequence'] = [fake_frame]
        assert_array_equal(MFW(fake_mf).image_orient_patient,
                        [[0, 1], [1, 0], [0, 0]])
        fake_mf['SharedFunctionalGroupsSequence'] = [None]
        assert_raises(didw.WrapperError,
                      getattr, MFW(fake_mf), 'image_orient_patient')
        fake_mf['PerFrameFunctionalGroupsSequence'] = [fake_frame]
        assert_array_equal(MFW(fake_mf).image_orient_patient,
                        [[0, 1], [1, 0], [0, 0]])

    def test_voxel_sizes(self):
        # Test voxel size calculation
        fake_mf = copy(self.MINIMAL_MF)
        MFW = self.WRAPCLASS
        dw = MFW(fake_mf)
        assert_raises(didw.WrapperError, getattr, dw, 'voxel_sizes')
        # Make a fake frame
        fake_frame = fake_frames('PixelMeasuresSequence',
                                 'PixelSpacing',
                                 [[2.1, 3.2]])[0]
        fake_mf['SharedFunctionalGroupsSequence'] = [fake_frame]
        # Still not enough, we lack information for slice distances
        assert_raises(didw.WrapperError, getattr, MFW(fake_mf), 'voxel_sizes')
        # This can come from SpacingBetweenSlices or frame SliceThickness
        fake_mf['SpacingBetweenSlices'] = 4.3
        assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 4.3])
        # If both, prefer SliceThickness
        fake_frame.PixelMeasuresSequence[0].SliceThickness = 5.4
        assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 5.4])
        # Just SliceThickness is OK
        del fake_mf['SpacingBetweenSlices']
        assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 5.4])
        # Removing shared leads to error again
        fake_mf['SharedFunctionalGroupsSequence'] = [None]
        assert_raises(didw.WrapperError, getattr, MFW(fake_mf), 'voxel_sizes')
        # Restoring to frames makes it work again
        fake_mf['PerFrameFunctionalGroupsSequence'] = [fake_frame]
        assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 5.4])
        # Decimals in any field are OK
        fake_frame = fake_frames('PixelMeasuresSequence',
                                 'PixelSpacing',
                                 [[Decimal('2.1'), Decimal('3.2')]])[0]
        fake_mf['SharedFunctionalGroupsSequence'] = [fake_frame]
        fake_mf['SpacingBetweenSlices'] = Decimal('4.3')
        assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 4.3])
        fake_frame.PixelMeasuresSequence[0].SliceThickness = Decimal('5.4')
        assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 5.4])

    def test_image_position(self):
        # Test image_position property for multiframe
        fake_mf = copy(self.MINIMAL_MF)
        MFW = self.WRAPCLASS
        dw = MFW(fake_mf)
        assert_raises(didw.WrapperError, getattr, dw, 'image_position')
        # Make a fake frame
        fake_frame = fake_frames('PlanePositionSequence',
                                 'ImagePositionPatient',
                                 [[-2.0, 3., 7]])[0]
        fake_mf['SharedFunctionalGroupsSequence'] = [fake_frame]
        assert_array_equal(MFW(fake_mf).image_position, [-2, 3, 7])
        fake_mf['SharedFunctionalGroupsSequence'] = [None]
        assert_raises(didw.WrapperError,
                    getattr, MFW(fake_mf), 'image_position')
        fake_mf['PerFrameFunctionalGroupsSequence'] = [fake_frame]
        assert_array_equal(MFW(fake_mf).image_position, [-2, 3, 7])
        # Check lists of Decimals work
        fake_frame.PlanePositionSequence[0].ImagePositionPatient = [
            Decimal(str(v)) for v in [-2, 3, 7]]
        assert_array_equal(MFW(fake_mf).image_position, [-2, 3, 7])
        assert_equal(MFW(fake_mf).image_position.dtype, float)

    @dicom_test
    def test_affine(self):
        # Make sure we find orientation/position/spacing info
        dw = didw.wrapper_from_file(DATA_FILE_4D)
        dw.get_affine()

    @dicom_test
    def test_data_real(self):
        # The data in this file is (initially) a 1D gradient so it compresses
        # well.  This just tests that the data ordering produces a consistent
        # result.
        dw = didw.wrapper_from_file(DATA_FILE_4D)
        dat_str = dw.get_data().tostring()
        assert_equal(sha1(dat_str).hexdigest(),
                    '149323269b0af92baa7508e19ca315240f77fa8c')

    def test_data_fake(self):
        # Test algorithm for get_data
        fake_mf = copy(self.MINIMAL_MF)
        MFW = self.WRAPCLASS
        dw = MFW(fake_mf)
        # Fails - no shape
        assert_raises(didw.WrapperError, dw.get_data)
        # Set shape by cheating
        dw.image_shape = (2, 3, 4)
        # Still fails - no data
        assert_raises(didw.WrapperError, dw.get_data)
        # Make shape and indices
        fake_mf['Rows'] = 2
        fake_mf['Columns'] = 3
        fake_mf['NumberOfFrames'] = 4
        frames = fake_frames('FrameContentSequence',
                             'DimensionIndexValues',
                             ((1, 1), (1, 2), (1, 3), (1, 4)))
        fake_mf['PerFrameFunctionalGroupsSequence'] = frames
        assert_equal(MFW(fake_mf).image_shape, (2, 3, 4))
        # Still fails - no data
        assert_raises(didw.WrapperError, dw.get_data)
        # Add data - 3D
        data = np.arange(24).reshape((2, 3, 4))
        # Frames dim is first for some reason
        fake_mf['pixel_array'] = np.rollaxis(data, 2)
        # Now it should work
        dw = MFW(fake_mf)
        assert_array_equal(dw.get_data(), data)
        # Test scaling works
        fake_mf['RescaleSlope'] = 2.0
        fake_mf['RescaleIntercept'] = -1
        assert_array_equal(MFW(fake_mf).get_data(), data * 2.0 - 1)
        # Check slice sorting
        frames = fake_frames('FrameContentSequence',
                             'DimensionIndexValues',
                             ((1, 4), (1, 2), (1, 3), (1, 1)))
        fake_mf['PerFrameFunctionalGroupsSequence'] = frames
        sorted_data = data[..., [3, 1, 2, 0]]
        fake_mf['pixel_array'] = np.rollaxis(sorted_data, 2)
        assert_array_equal(MFW(fake_mf).get_data(), data * 2.0 - 1)
        # 5D!
        dim_idxs = [
            [1, 4, 2, 1],
            [1, 2, 2, 1],
            [1, 3, 2, 1],
            [1, 1, 2, 1],
            [1, 4, 2, 2],
            [1, 2, 2, 2],
            [1, 3, 2, 2],
            [1, 1, 2, 2],
            [1, 4, 1, 1],
            [1, 2, 1, 1],
            [1, 3, 1, 1],
            [1, 1, 1, 1],
            [1, 4, 1, 2],
            [1, 2, 1, 2],
            [1, 3, 1, 2],
            [1, 1, 1, 2]]
        frames = fake_frames('FrameContentSequence',
                             'DimensionIndexValues',
                             dim_idxs)
        fake_mf['PerFrameFunctionalGroupsSequence'] = frames
        fake_mf['NumberOfFrames'] = len(frames)
        shape = (2, 3, 4, 2, 2)
        data = np.arange(np.prod(shape)).reshape(shape)
        sorted_data = data.reshape(shape[:2] + (-1,), order='F')
        order = [11,  9, 10,  8,  3,  1,  2,  0,
                 15, 13, 14, 12,  7,  5,  6,  4]
        sorted_data = sorted_data[..., np.argsort(order)]
        fake_mf['pixel_array'] = np.rollaxis(sorted_data, 2)
        assert_array_equal(MFW(fake_mf).get_data(), data * 2.0 - 1)

    def test__scale_data(self):
        # Test data scaling
        fake_mf = copy(self.MINIMAL_MF)
        MFW = self.WRAPCLASS
        dw = MFW(fake_mf)
        data = np.arange(24).reshape((2, 3, 4))
        assert_array_equal(data, dw._scale_data(data))
        fake_mf['RescaleSlope'] = 2.0
        fake_mf['RescaleIntercept'] = -1.0
        assert_array_equal(data * 2 - 1, dw._scale_data(data))
        fake_frame = fake_frames('PixelValueTransformationSequence',
                                 'RescaleSlope',
                                 [3.0])[0]
        fake_mf['PerFrameFunctionalGroupsSequence'] = [fake_frame]
        # Lacking RescaleIntercept -> Error
        dw = MFW(fake_mf)
        assert_raises(AttributeError, dw._scale_data, data)
        fake_frame.PixelValueTransformationSequence[0].RescaleIntercept = -2
        assert_array_equal(data * 3 - 2, dw._scale_data(data))
        # Decimals are OK
        fake_frame.PixelValueTransformationSequence[0].RescaleSlope = Decimal('3')
        fake_frame.PixelValueTransformationSequence[0].RescaleIntercept = Decimal('-2')
        assert_array_equal(data * 3 - 2, dw._scale_data(data))
