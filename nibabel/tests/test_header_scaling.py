""" Testing header scaling

"""

import numpy as np

import nibabel as nib
from nibabel.spatialimages import HeaderDataError, HeaderTypeError

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nibabel.testing import parametric


@parametric
def test_hdr_scaling():
    hdr = nib.AnalyzeHeader()
    hdr.set_data_dtype(np.float32)
    data = np.arange(6, dtype=np.float32).reshape(1,2,3)
    res = hdr.scaling_from_data(data)
    yield assert_equal(res, (1.0, 0.0, None, None))
     # The Analyze header cannot scale
    hdr.set_data_dtype(np.uint8)
    yield assert_raises(HeaderTypeError, hdr.scaling_from_data, data)
     # The nifti header can scale
    hdr = nib.Nifti1Header()
    hdr.set_data_dtype(np.uint8)
    slope, inter, mx, mn = hdr.scaling_from_data(data)
    yield assert_equal((inter, mx, mn), (0.0, 0, 5))
    yield assert_array_almost_equal(slope, 5.0/255)

