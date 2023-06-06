"""Testing DICOM wrappers
"""

import gzip
from copy import copy
from decimal import Decimal
from hashlib import sha1
from os.path import dirname
from os.path import join as pjoin
from unittest import TestCase

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ...tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from ...volumeutils import endian_codes
from .. import dicomreaders as didr
from .. import dicomwrappers as didw
from . import dicom_test, have_dicom, pydicom

IO_DATA_PATH = pjoin(dirname(__file__), 'data')
DATA_FILE = pjoin(IO_DATA_PATH, 'siemens_dwi_1000.dcm.gz')
DATA_FILE_PHILIPS = pjoin(IO_DATA_PATH, 'philips_mprage.dcm.gz')
if have_dicom:
    DATA = pydicom.read_file(gzip.open(DATA_FILE))
    DATA_PHILIPS = pydicom.read_file(gzip.open(DATA_FILE_PHILIPS))
else:
    DATA = None
    DATA_PHILIPS = None
DATA_FILE_B0 = pjoin(IO_DATA_PATH, 'siemens_dwi_0.dcm.gz')
DATA_FILE_SLC_NORM = pjoin(IO_DATA_PATH, 'csa_slice_norm.dcm')
DATA_FILE_DEC_RSCL = pjoin(IO_DATA_PATH, 'decimal_rescale.dcm')
DATA_FILE_4D = pjoin(IO_DATA_PATH, '4d_multiframe_test.dcm')
DATA_FILE_EMPTY_ST = pjoin(IO_DATA_PATH, 'slicethickness_empty_string.dcm')
DATA_FILE_4D_DERIVED = pjoin(get_nibabel_data(), 'nitest-dicom', '4d_multiframe_with_derived.dcm')
DATA_FILE_CT = pjoin(get_nibabel_data(), 'nitest-dicom', 'siemens_ct_header_csa.dcm')

# This affine from our converted image was shown to match our image spatially
# with an image from SPM DICOM conversion. We checked the matching with SPM
# check reg.  We have flipped the first and second rows to allow for rows, cols
# transpose in current return compared to original case.
EXPECTED_AFFINE = np.array(  # do this for philips?
    [
        [-1.796875, 0, 0, 115],
        [0, -1.79684984, -0.01570896, 135.028779],
        [0, -0.00940843750, 2.99995887, -78.710481],
        [0, 0, 0, 1],
    ]
)[:, [1, 0, 2, 3]]

# from Guys and Matthew's SPM code, undoing SPM's Y flip, and swapping first two
# values in vector, to account for data rows, cols difference.
EXPECTED_PARAMS = [992.05050247, (0.00507649, 0.99997450, -0.005023611)]


@dicom_test
def test_wrappers():
    # test direct wrapper calls
    # first with empty or minimal data
    multi_minimal = {
        'PerFrameFunctionalGroupsSequence': [None],
        'SharedFunctionalGroupsSequence': [None],
    }
    for maker, args in (
        (didw.Wrapper, ({},)),
        (didw.SiemensWrapper, ({},)),
        (didw.MosaicWrapper, ({}, None, 10)),
        (didw.MultiframeWrapper, (multi_minimal,)),
    ):
        dw = maker(*args)
        assert dw.get('InstanceNumber') is None
        assert dw.get('AcquisitionNumber') is None
        with pytest.raises(KeyError):
            dw['not an item']
        with pytest.raises(didw.WrapperError):
            dw.get_data()
        with pytest.raises(didw.WrapperError):
            dw.affine
        with pytest.raises(TypeError):
            maker()
        # Check default attributes
        if not maker is didw.MosaicWrapper:
            assert not dw.is_mosaic
        assert dw.b_matrix is None
        assert dw.q_vector is None
    for maker in (didw.wrapper_from_data, didw.Wrapper, didw.SiemensWrapper, didw.MosaicWrapper):
        dw = maker(DATA)
        assert dw.get('InstanceNumber') == 2
        assert dw.get('AcquisitionNumber') == 2
        with pytest.raises(KeyError):
            dw['not an item']
    for maker in (didw.MosaicWrapper, didw.wrapper_from_data):
        dw = maker(DATA)
        assert dw.is_mosaic
    # DATA is not a Multiframe DICOM file
    with pytest.raises(didw.WrapperError):
        didw.MultiframeWrapper(DATA)


def test_get_from_wrapper():
    # Test that 'get', and __getitem__ work as expected for underlying dicom
    # data
    dcm_data = {'some_key': 'some value'}
    dw = didw.Wrapper(dcm_data)
    assert dw.get('some_key') == 'some value'
    assert dw.get('some_other_key') is None
    # Getitem uses the same dictionary access
    assert dw['some_key'] == 'some value'
    # And raises a WrapperError for missing keys
    with pytest.raises(KeyError):
        dw['some_other_key']
    # Test we don't use attributes for get

    class FakeData(dict):
        pass

    d = FakeData()
    d.some_key = 'another bit of data'
    dw = didw.Wrapper(d)
    assert dw.get('some_key') is None
    # Check get defers to dcm_data get

    class FakeData2:
        def get(self, key, default):
            return 1

    d = FakeData2()
    d.some_key = 'another bit of data'
    dw = didw.Wrapper(d)
    assert dw.get('some_key') == 1


@dicom_test
def test_wrapper_from_data():
    # test wrapper from data, wrapper from file
    for dw in (didw.wrapper_from_data(DATA), didw.wrapper_from_file(DATA_FILE)):
        assert dw.get('InstanceNumber') == 2
        assert dw.get('AcquisitionNumber') == 2
        with pytest.raises(KeyError):
            dw['not an item']
        assert dw.is_mosaic
        assert_array_almost_equal(np.dot(didr.DPCS_TO_TAL, dw.affine), EXPECTED_AFFINE)
    for dw in (didw.wrapper_from_data(DATA_PHILIPS), didw.wrapper_from_file(DATA_FILE_PHILIPS)):
        assert dw.get('InstanceNumber') == 1
        assert dw.get('AcquisitionNumber') == 3
        with pytest.raises(KeyError):
            dw['not an item']
        assert dw.is_multiframe
    # Another CSA file
    dw = didw.wrapper_from_file(DATA_FILE_SLC_NORM)
    assert dw.is_mosaic
    # Check that multiframe requires minimal set of DICOM tags
    fake_data = dict()
    fake_data['SOPClassUID'] = '1.2.840.10008.5.1.4.1.1.4.2'
    dw = didw.wrapper_from_data(fake_data)
    assert not dw.is_multiframe
    # use the correct SOPClassUID
    fake_data['SOPClassUID'] = '1.2.840.10008.5.1.4.1.1.4.1'
    with pytest.raises(didw.WrapperError):
        didw.wrapper_from_data(fake_data)
    fake_data['PerFrameFunctionalGroupsSequence'] = [None]
    with pytest.raises(didw.WrapperError):
        didw.wrapper_from_data(fake_data)
    fake_data['SharedFunctionalGroupsSequence'] = [None]
    # minimal set should now be met
    dw = didw.wrapper_from_data(fake_data)
    assert dw.is_multiframe


@dicom_test
def test_wrapper_args_kwds():
    # Test we can pass args, kwargs to read_file
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
    with pytest.raises(pydicom.filereader.InvalidDicomError):
        didw.wrapper_from_file(csa_fname)
    # We can force the read, in which case rubbish returns
    dcm_malo = didw.wrapper_from_file(csa_fname, force=True)
    assert not dcm_malo.is_mosaic


@dicom_test
def test_dwi_params():
    dw = didw.wrapper_from_data(DATA)
    b_matrix = dw.b_matrix
    assert b_matrix.shape == (3, 3)
    q = dw.q_vector
    b = np.sqrt(np.sum(q * q))  # vector norm
    g = q / b
    assert_array_almost_equal(b, EXPECTED_PARAMS[0])
    assert_array_almost_equal(g, EXPECTED_PARAMS[1])


@dicom_test
def test_q_vector_etc():
    # Test diffusion params in wrapper classes
    # Default is no q_vector, b_value, b_vector
    dw = didw.Wrapper(DATA)
    assert dw.q_vector is None
    assert dw.b_value is None
    assert dw.b_vector is None
    for pos in range(3):
        q_vec = np.zeros((3,))
        q_vec[pos] = 10.0
        # Reset wrapped dicom to refresh one_time property
        dw = didw.Wrapper(DATA)
        dw.q_vector = q_vec
        assert_array_equal(dw.q_vector, q_vec)
        assert dw.b_value == 10
        assert_array_equal(dw.b_vector, q_vec / 10.0)
    # Reset wrapped dicom to refresh one_time property
    dw = didw.Wrapper(DATA)
    dw.q_vector = np.array([0, 0, 1e-6])
    assert dw.b_value == 0
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
    assert sdw.b_value == 0
    assert_array_equal(sdw.b_vector, np.zeros((3,)))


@dicom_test
def test_vol_matching():
    # make the Siemens wrapper, check it compares True against itself
    dw_siemens = didw.wrapper_from_data(DATA)
    assert dw_siemens.is_mosaic
    assert dw_siemens.is_csa
    assert dw_siemens.is_same_series(dw_siemens)
    # make plain wrapper, compare against itself
    dw_plain = didw.Wrapper(DATA)
    assert not dw_plain.is_mosaic
    assert not dw_plain.is_csa
    assert dw_plain.is_same_series(dw_plain)
    # specific vs plain wrapper compares False, because the Siemens
    # wrapper has more non-empty information
    assert not dw_plain.is_same_series(dw_siemens)
    # and this should be symmetric
    assert not dw_siemens.is_same_series(dw_plain)
    # we can even make an empty wrapper.  This compares True against
    # itself but False against the others
    dw_empty = didw.Wrapper({})
    assert dw_empty.is_same_series(dw_empty)
    assert not dw_empty.is_same_series(dw_plain)
    assert not dw_plain.is_same_series(dw_empty)
    # Just to check the interface, make a pretend signature-providing
    # object.

    class C:
        series_signature = {}

    assert dw_empty.is_same_series(C())

    # make the Philips wrapper, check it compares True against itself
    dw_philips = didw.wrapper_from_data(DATA_PHILIPS)
    assert dw_philips.is_multiframe
    assert dw_philips.is_same_series(dw_philips)
    # make plain wrapper, compare against itself
    dw_plain_philips = didw.Wrapper(DATA)
    assert not dw_plain_philips.is_multiframe
    assert dw_plain_philips.is_same_series(dw_plain_philips)
    # specific vs plain wrapper compares False, because the Philips
    # wrapper has more non-empty information
    assert not dw_plain_philips.is_same_series(dw_philips)
    # and this should be symmetric
    assert not dw_philips.is_same_series(dw_plain_philips)
    # we can even make an empty wrapper.  This compares True against
    # itself but False against the others
    dw_empty = didw.Wrapper({})
    assert dw_empty.is_same_series(dw_empty)
    assert not dw_empty.is_same_series(dw_plain_philips)
    assert not dw_plain_philips.is_same_series(dw_empty)


@dicom_test
def test_slice_indicator():
    dw_0 = didw.wrapper_from_file(DATA_FILE_B0)
    dw_1000 = didw.wrapper_from_data(DATA)
    z = dw_0.slice_indicator
    assert not z is None
    assert z == dw_1000.slice_indicator
    dw_empty = didw.Wrapper({})
    assert dw_empty.slice_indicator is None


@dicom_test
def test_orthogonal():
    # Test that the slice normal is sufficiently orthogonal
    dw = didw.wrapper_from_file(DATA_FILE_SLC_NORM)
    R = dw.rotation_matrix
    assert np.allclose(np.eye(3), np.dot(R, R.T), atol=1e-6)

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
    with pytest.raises(didw.WrapperPrecisionError):
        dw.rotation_matrix


@dicom_test
def test_rotation_matrix():
    # Test rotation matrix and slice normal
    d = {}
    d['ImageOrientationPatient'] = [0, 1, 0, 1, 0, 0]
    dw = didw.wrapper_from_data(d)
    assert_array_equal(dw.rotation_matrix, np.eye(3))
    d['ImageOrientationPatient'] = [1, 0, 0, 0, 1, 0]
    dw = didw.wrapper_from_data(d)
    assert_array_equal(dw.rotation_matrix, [[0, 1, 0], [1, 0, 0], [0, 0, -1]])


@dicom_test
def test_use_csa_sign():
    # Test that we get the same slice normal, even after swapping the iop
    # directions
    dw = didw.wrapper_from_file(DATA_FILE_SLC_NORM)
    iop = dw.image_orient_patient
    dw.image_orient_patient = np.c_[iop[:, 1], iop[:, 0]]
    dw2 = didw.wrapper_from_file(DATA_FILE_SLC_NORM)
    assert np.allclose(dw.slice_normal, dw2.slice_normal)


@dicom_test
def test_assert_parallel():
    # Test that we get an AssertionError if the cross product and the CSA
    # slice normal are not parallel
    dw = didw.wrapper_from_file(DATA_FILE_SLC_NORM)
    dw.image_orient_patient = np.c_[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    with pytest.raises(AssertionError):
        dw.slice_normal


@dicom_test
def test_decimal_rescale():
    # Test that we don't get back a data array with dtype object when our
    # rescale slope is a decimal
    dw = didw.wrapper_from_file(DATA_FILE_DEC_RSCL)
    assert dw.get_data().dtype != np.dtype(object)


def fake_frames(seq_name, field_name, value_seq):
    """Make fake frames for multiframe testing

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

    class Fake:
        pass

    frames = []
    for value in value_seq:
        fake_frame = Fake()
        fake_element = Fake()
        setattr(fake_element, field_name, value)
        setattr(fake_frame, seq_name, [fake_element])
        frames.append(fake_frame)
    return frames


def fake_shape_dependents(div_seq, sid_seq=None, sid_dim=None):
    """Make a fake dictionary of data that ``image_shape`` is dependent on.

    Parameters
    ----------
    div_seq : list of tuples
        list of values to use for the `DimensionIndexValues` of each frame.
    sid_seq : list of int
        list of values to use for the `StackID` of each frame.
    sid_dim : int
        the index of the column in 'div_seq' to use as 'sid_seq'
    """

    class DimIdxSeqElem:
        def __init__(self, dip=(0, 0), fgp=None):
            self.DimensionIndexPointer = dip
            if fgp is not None:
                self.FunctionalGroupPointer = fgp

    class FrmContSeqElem:
        def __init__(self, div, sid):
            self.DimensionIndexValues = div
            self.StackID = sid

    class PerFrmFuncGrpSeqElem:
        def __init__(self, div, sid):
            self.FrameContentSequence = [FrmContSeqElem(div, sid)]

    # if no StackID values passed in then use the values at index 'sid_dim' in
    # the value for DimensionIndexValues for it
    if sid_seq is None:
        if sid_dim is None:
            sid_dim = 0
        sid_seq = [div[sid_dim] for div in div_seq]
    # create the DimensionIndexSequence
    num_of_frames = len(div_seq)
    dim_idx_seq = [DimIdxSeqElem()] * num_of_frames
    # add an entry for StackID into the DimensionIndexSequence
    if sid_dim is not None:
        sid_tag = pydicom.datadict.tag_for_keyword('StackID')
        fcs_tag = pydicom.datadict.tag_for_keyword('FrameContentSequence')
        dim_idx_seq[sid_dim] = DimIdxSeqElem(sid_tag, fcs_tag)
    # create the PerFrameFunctionalGroupsSequence
    frames = [PerFrmFuncGrpSeqElem(div, sid) for div, sid in zip(div_seq, sid_seq)]
    return {
        'NumberOfFrames': num_of_frames,
        'DimensionIndexSequence': dim_idx_seq,
        'PerFrameFunctionalGroupsSequence': frames,
    }


class TestMultiFrameWrapper(TestCase):
    # Test MultiframeWrapper
    MINIMAL_MF = {
        # Minimal contents of dcm_data for this wrapper
        'PerFrameFunctionalGroupsSequence': [None],
        'SharedFunctionalGroupsSequence': [None],
    }
    WRAPCLASS = didw.MultiframeWrapper

    @dicom_test
    def test_shape(self):
        # Check the shape algorithm
        fake_mf = copy(self.MINIMAL_MF)
        MFW = self.WRAPCLASS
        dw = MFW(fake_mf)
        # No rows, cols, raise WrapperError
        with pytest.raises(didw.WrapperError):
            dw.image_shape
        fake_mf['Rows'] = 64
        with pytest.raises(didw.WrapperError):
            dw.image_shape
        fake_mf.pop('Rows')
        fake_mf['Columns'] = 64
        with pytest.raises(didw.WrapperError):
            dw.image_shape
        fake_mf['Rows'] = 32
        # Missing frame data, raise AssertionError
        with pytest.raises(AssertionError):
            dw.image_shape
        fake_mf['NumberOfFrames'] = 4
        # PerFrameFunctionalGroupsSequence does not match NumberOfFrames
        with pytest.raises(AssertionError):
            dw.image_shape
        # check 3D shape when StackID index is 0
        div_seq = ((1, 1), (1, 2), (1, 3), (1, 4))
        fake_mf.update(fake_shape_dependents(div_seq, sid_dim=0))
        assert MFW(fake_mf).image_shape == (32, 64, 4)
        # Check stack number matching when StackID index is 0
        div_seq = ((1, 1), (1, 2), (1, 3), (2, 4))
        fake_mf.update(fake_shape_dependents(div_seq, sid_dim=0))
        with pytest.raises(didw.WrapperError):
            MFW(fake_mf).image_shape
        # Make some fake frame data for 4D when StackID index is 0
        div_seq = ((1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 2, 2), (1, 1, 3), (1, 2, 3))
        fake_mf.update(fake_shape_dependents(div_seq, sid_dim=0))
        assert MFW(fake_mf).image_shape == (32, 64, 2, 3)
        # Check stack number matching for 4D when StackID index is 0
        div_seq = ((1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 2, 2), (1, 1, 3), (2, 2, 3))
        fake_mf.update(fake_shape_dependents(div_seq, sid_dim=0))
        with pytest.raises(didw.WrapperError):
            MFW(fake_mf).image_shape
        # Check indices can be non-contiguous when StackID index is 0
        div_seq = ((1, 1, 1), (1, 2, 1), (1, 1, 3), (1, 2, 3))
        fake_mf.update(fake_shape_dependents(div_seq, sid_dim=0))
        assert MFW(fake_mf).image_shape == (32, 64, 2, 2)
        # Check indices can include zero when StackID index is 0
        div_seq = ((1, 1, 0), (1, 2, 0), (1, 1, 3), (1, 2, 3))
        fake_mf.update(fake_shape_dependents(div_seq, sid_dim=0))
        assert MFW(fake_mf).image_shape == (32, 64, 2, 2)
        # check 3D shape when there is no StackID index
        div_seq = ((1,), (2,), (3,), (4,))
        sid_seq = (1, 1, 1, 1)
        fake_mf.update(fake_shape_dependents(div_seq, sid_seq=sid_seq))
        assert MFW(fake_mf).image_shape == (32, 64, 4)
        # check 3D stack number matching when there is no StackID index
        div_seq = ((1,), (2,), (3,), (4,))
        sid_seq = (1, 1, 1, 2)
        fake_mf.update(fake_shape_dependents(div_seq, sid_seq=sid_seq))
        with pytest.raises(didw.WrapperError):
            MFW(fake_mf).image_shape
        # check 4D shape when there is no StackID index
        div_seq = ((1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3))
        sid_seq = (1, 1, 1, 1, 1, 1)
        fake_mf.update(fake_shape_dependents(div_seq, sid_seq=sid_seq))
        assert MFW(fake_mf).image_shape == (32, 64, 2, 3)
        # check 4D stack number matching when there is no StackID index
        div_seq = ((1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3))
        sid_seq = (1, 1, 1, 1, 1, 2)
        fake_mf.update(fake_shape_dependents(div_seq, sid_seq=sid_seq))
        with pytest.raises(didw.WrapperError):
            MFW(fake_mf).image_shape
        # check 3D shape when StackID index is 1
        div_seq = ((1, 1), (2, 1), (3, 1), (4, 1))
        fake_mf.update(fake_shape_dependents(div_seq, sid_dim=1))
        assert MFW(fake_mf).image_shape == (32, 64, 4)
        # Check stack number matching when StackID index is 1
        div_seq = ((1, 1), (2, 1), (3, 2), (4, 1))
        fake_mf.update(fake_shape_dependents(div_seq, sid_dim=1))
        with pytest.raises(didw.WrapperError):
            MFW(fake_mf).image_shape
        # Make some fake frame data for 4D when StackID index is 1
        div_seq = ((1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2), (1, 1, 3), (2, 1, 3))
        fake_mf.update(fake_shape_dependents(div_seq, sid_dim=1))
        assert MFW(fake_mf).image_shape == (32, 64, 2, 3)

    def test_iop(self):
        # Test Image orient patient for multiframe
        fake_mf = copy(self.MINIMAL_MF)
        MFW = self.WRAPCLASS
        dw = MFW(fake_mf)
        with pytest.raises(didw.WrapperError):
            dw.image_orient_patient
        # Make a fake frame
        fake_frame = fake_frames(
            'PlaneOrientationSequence', 'ImageOrientationPatient', [[0, 1, 0, 1, 0, 0]]
        )[0]
        fake_mf['SharedFunctionalGroupsSequence'] = [fake_frame]
        assert_array_equal(MFW(fake_mf).image_orient_patient, [[0, 1], [1, 0], [0, 0]])
        fake_mf['SharedFunctionalGroupsSequence'] = [None]
        with pytest.raises(didw.WrapperError):
            MFW(fake_mf).image_orient_patient
        fake_mf['PerFrameFunctionalGroupsSequence'] = [fake_frame]
        assert_array_equal(MFW(fake_mf).image_orient_patient, [[0, 1], [1, 0], [0, 0]])

    def test_voxel_sizes(self):
        # Test voxel size calculation
        fake_mf = copy(self.MINIMAL_MF)
        MFW = self.WRAPCLASS
        dw = MFW(fake_mf)
        with pytest.raises(didw.WrapperError):
            dw.voxel_sizes
        # Make a fake frame
        fake_frame = fake_frames('PixelMeasuresSequence', 'PixelSpacing', [[2.1, 3.2]])[0]
        fake_mf['SharedFunctionalGroupsSequence'] = [fake_frame]
        # Still not enough, we lack information for slice distances
        with pytest.raises(didw.WrapperError):
            MFW(fake_mf).voxel_sizes
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
        with pytest.raises(didw.WrapperError):
            MFW(fake_mf).voxel_sizes
        # Restoring to frames makes it work again
        fake_mf['PerFrameFunctionalGroupsSequence'] = [fake_frame]
        assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 5.4])
        # Decimals in any field are OK
        fake_frame = fake_frames(
            'PixelMeasuresSequence', 'PixelSpacing', [[Decimal('2.1'), Decimal('3.2')]]
        )[0]
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
        with pytest.raises(didw.WrapperError):
            dw.image_position
        # Make a fake frame
        fake_frame = fake_frames(
            'PlanePositionSequence', 'ImagePositionPatient', [[-2.0, 3.0, 7]]
        )[0]
        fake_mf['SharedFunctionalGroupsSequence'] = [fake_frame]
        assert_array_equal(MFW(fake_mf).image_position, [-2, 3, 7])
        fake_mf['SharedFunctionalGroupsSequence'] = [None]
        with pytest.raises(didw.WrapperError):
            MFW(fake_mf).image_position
        fake_mf['PerFrameFunctionalGroupsSequence'] = [fake_frame]
        assert_array_equal(MFW(fake_mf).image_position, [-2, 3, 7])
        # Check lists of Decimals work
        fake_frame.PlanePositionSequence[0].ImagePositionPatient = [
            Decimal(str(v)) for v in [-2, 3, 7]
        ]
        assert_array_equal(MFW(fake_mf).image_position, [-2, 3, 7])
        assert MFW(fake_mf).image_position.dtype == float

    @dicom_test
    @pytest.mark.xfail(reason='Not packaged in install', raises=FileNotFoundError)
    def test_affine(self):
        # Make sure we find orientation/position/spacing info
        dw = didw.wrapper_from_file(DATA_FILE_4D)
        aff = dw.affine

    @dicom_test
    @pytest.mark.xfail(reason='Not packaged in install', raises=FileNotFoundError)
    def test_data_real(self):
        # The data in this file is (initially) a 1D gradient so it compresses
        # well.  This just tests that the data ordering produces a consistent
        # result.
        dw = didw.wrapper_from_file(DATA_FILE_4D)
        data = dw.get_data()
        # data hash depends on the endianness
        if endian_codes[data.dtype.byteorder] == '>':
            data = data.byteswap()
        dat_str = data.tobytes()
        assert sha1(dat_str).hexdigest() == '149323269b0af92baa7508e19ca315240f77fa8c'

    @dicom_test
    def test_slicethickness_fallback(self):
        dw = didw.wrapper_from_file(DATA_FILE_EMPTY_ST)
        assert dw.voxel_sizes[2] == 1.0

    @dicom_test
    @needs_nibabel_data('nitest-dicom')
    def test_data_derived_shape(self):
        # Test 4D diffusion data with an additional trace volume included
        # Excludes the trace volume and generates the correct shape
        dw = didw.wrapper_from_file(DATA_FILE_4D_DERIVED)
        with pytest.warns(UserWarning, match='Derived images found and removed'):
            assert dw.image_shape == (96, 96, 60, 33)

    @dicom_test
    @needs_nibabel_data('nitest-dicom')
    def test_data_unreadable_private_headers(self):
        # Test CT image with unreadable CSA tags
        with pytest.warns(UserWarning, match='Error while attempting to read CSA header'):
            dw = didw.wrapper_from_file(DATA_FILE_CT)
        assert dw.image_shape == (512, 571)

    @dicom_test
    def test_data_fake(self):
        # Test algorithm for get_data
        fake_mf = copy(self.MINIMAL_MF)
        MFW = self.WRAPCLASS
        dw = MFW(fake_mf)
        # Fails - no shape
        with pytest.raises(didw.WrapperError):
            dw.get_data()
        # Set shape by cheating
        dw.image_shape = (2, 3, 4)
        # Still fails - no data
        with pytest.raises(didw.WrapperError):
            dw.get_data()
        # Make shape and indices
        fake_mf['Rows'] = 2
        fake_mf['Columns'] = 3
        dim_idxs = ((1, 1), (1, 2), (1, 3), (1, 4))
        fake_mf.update(fake_shape_dependents(dim_idxs, sid_dim=0))
        assert MFW(fake_mf).image_shape == (2, 3, 4)
        # Still fails - no data
        with pytest.raises(didw.WrapperError):
            dw.get_data()
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
        dim_idxs = ((1, 4), (1, 2), (1, 3), (1, 1))
        fake_mf.update(fake_shape_dependents(dim_idxs, sid_dim=0))
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
            [1, 1, 1, 2],
        ]
        fake_mf.update(fake_shape_dependents(dim_idxs, sid_dim=0))
        shape = (2, 3, 4, 2, 2)
        data = np.arange(np.prod(shape)).reshape(shape)
        sorted_data = data.reshape(shape[:2] + (-1,), order='F')
        order = [11, 9, 10, 8, 3, 1, 2, 0, 15, 13, 14, 12, 7, 5, 6, 4]
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
        fake_frame = fake_frames('PixelValueTransformationSequence', 'RescaleSlope', [3.0])[0]
        fake_mf['PerFrameFunctionalGroupsSequence'] = [fake_frame]
        # Lacking RescaleIntercept -> Error
        dw = MFW(fake_mf)
        with pytest.raises(AttributeError):
            dw._scale_data(data)
        fake_frame.PixelValueTransformationSequence[0].RescaleIntercept = -2
        assert_array_equal(data * 3 - 2, dw._scale_data(data))
        # Decimals are OK
        fake_frame.PixelValueTransformationSequence[0].RescaleSlope = Decimal('3')
        fake_frame.PixelValueTransformationSequence[0].RescaleIntercept = Decimal('-2')
        assert_array_equal(data * 3 - 2, dw._scale_data(data))
