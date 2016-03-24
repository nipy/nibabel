""" Testing reading DICOM files

"""

import numpy as np

from .. import dicomreaders as didr

from nibabel.pydicom_compat import dicom_test, pydicom

from .test_dicomwrappers import (EXPECTED_AFFINE,
                                 EXPECTED_PARAMS,
                                 IO_DATA_PATH,
                                 DATA)

from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)

from numpy.testing import assert_array_equal, assert_array_almost_equal


@dicom_test
def test_read_dwi():
    img = didr.mosaic_to_nii(DATA)
    arr = img.get_data()
    assert_equal(arr.shape, (128, 128, 48))
    assert_array_almost_equal(img.affine, EXPECTED_AFFINE)


@dicom_test
def test_read_dwis():
    data, aff, bs, gs = didr.read_mosaic_dwi_dir(IO_DATA_PATH,
                                                 'siemens_dwi_*.dcm.gz')
    assert_equal(data.ndim, 4)
    assert_array_almost_equal(aff, EXPECTED_AFFINE)
    assert_array_almost_equal(bs, (0, EXPECTED_PARAMS[0]))
    assert_array_almost_equal(gs, (np.zeros((3,)), EXPECTED_PARAMS[1]))
    assert_raises(IOError, didr.read_mosaic_dwi_dir, 'improbable')


@dicom_test
def test_passing_kwds():
    # Check that we correctly pass keywords to dicom
    dwi_glob = 'siemens_dwi_*.dcm.gz'
    csa_glob = 'csa*.bin'
    for func in (didr.read_mosaic_dwi_dir, didr.read_mosaic_dir):
        data, aff, bs, gs = func(IO_DATA_PATH, dwi_glob)
        # This should not raise an error
        data2, aff2, bs2, gs2 = func(
            IO_DATA_PATH,
            dwi_glob,
            dicom_kwargs=dict(force=True))
        assert_array_equal(data, data2)
        # This should raise an error in pydicom.dicomio.read_file
        assert_raises(TypeError,
                      func,
                      IO_DATA_PATH,
                      dwi_glob,
                      dicom_kwargs=dict(not_a_parameter=True))
        # These are invalid dicoms, so will raise an error unless force=True
        assert_raises(pydicom.filereader.InvalidDicomError,
                      func,
                      IO_DATA_PATH,
                      csa_glob)
        # But here, we catch the error because the dicoms are in the wrong
        # format
        assert_raises(didr.DicomReadError,
                      func,
                      IO_DATA_PATH,
                      csa_glob,
                      dicom_kwargs=dict(force=True))
