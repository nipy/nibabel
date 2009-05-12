import os
import urllib
import tempfile

import numpy as np

from scipy.io.netcdf import netcdf_file as netcdf

import nifti.minc as minc
reload(minc)

from nose.tools import assert_true, assert_equal, assert_false
from numpy.testing import assert_array_equal


def setup_module():
    global _fname, mnc
    fd, _fname = tempfile.mkstemp('.mnc')
    url = 'http://cirl.berkeley.edu/mb312/example_images/minc/avg152T1.mnc'
    urllib.urlretrieve(url, _fname)
    mnc = minc.MINCHeader(netcdf(_fname, 'r'))
    

def teardown_module():
    os.unlink(_fname)


def test_eg_img():
    yield assert_equal, mnc.get_data_dtype().type, np.uint8
    yield assert_equal, mnc.get_data_shape(), (91,109, 91)
    yield assert_equal, mnc.get_zooms(), (2.0, 2.0, 2.0)
    aff = np.array([[0, 0, 2.0, -90],
                    [0, 2.0, 0, -126],
                    [2.0, 0, 0, -72],
                    [0, 0, 0, 1]])
    yield assert_array_equal, mnc.get_best_affine(), aff
    data = mnc.get_unscaled_data()
    yield assert_equal, data.shape, (91,109, 91)
    data = mnc.get_scaled_data()
    yield assert_equal, data.shape, (91,109, 91)
    img = minc.load(_fname)
    data = img.get_data()
    yield assert_equal, data.shape, (91,109, 91)
    yield assert_equal, data.min(), 0.0
    yield assert_equal, data.max(), 1.0
