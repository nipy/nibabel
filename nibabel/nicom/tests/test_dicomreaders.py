""" Testing reading DICOM files

"""

import numpy as np

from .. import dicomreaders as didr

from .test_dicomwrappers import (dicom_test,
                                 EXPECTED_AFFINE,
                                 EXPECTED_PARAMS,
                                 IO_DATA_PATH,
                                 DATA)

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

@dicom_test
def test_read_dwi():
    img = didr.mosaic_to_nii(DATA)
    arr = img.get_data()
    assert_equal(arr.shape, (128,128,48))
    assert_array_almost_equal(img.get_affine(), EXPECTED_AFFINE)


@dicom_test
def test_read_dwis():
    data, aff, bs, gs = didr.read_mosaic_dwi_dir(IO_DATA_PATH, 
                                                 'siemens_dwi_*.dcm.gz')
    assert_equal(data.ndim, 4)
    assert_array_almost_equal(aff, EXPECTED_AFFINE)
    assert_array_almost_equal(bs, (0, EXPECTED_PARAMS[0]))
    assert_array_almost_equal(gs,
                              (np.zeros((3,)) + np.nan,
                               EXPECTED_PARAMS[1]))
    assert_raises(IOError, didr.read_mosaic_dwi_dir, 'improbable')
