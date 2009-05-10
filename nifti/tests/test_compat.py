''' Tests for pynifti compat package '''
import os
import numpy as np

from numpy.testing import assert_array_equal
from nose.tools import ok_

from nifti.volumeutils import HeaderDataError
from nifti.compat import NiftiImage

data_path, _ = os.path.split(__file__)
data_path = os.path.join(data_path, 'data')
image_file = os.path.join(data_path, 'example4d.nii.gz')

data_shape = (2, 24, 96, 128)

def test_theverybasics():
    nim = NiftiImage(image_file)
    ok_(nim.getDataArray().shape == data_shape)
