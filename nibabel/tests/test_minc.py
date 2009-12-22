from os.path import join as pjoin

import numpy as np
import numpy.testing.decorators as dec

from scipy.io.netcdf import netcdf_file as netcdf

from nibabel import load, MincHeader, Nifti1Image

from nose.tools import assert_true, assert_equal, assert_false
from numpy.testing import assert_array_equal

try:
    import imagedata
except ImportError:
    decimg = dec.skipif(True, 'no imagedata package on python path')
else:
    decimg = lambda x : x
    

@decimg
def test_eg_img():
    mnc_fname = pjoin(imagedata.minc_path, 'avg152T1.mnc')
    mnc = MincHeader(netcdf(mnc_fname, 'r'))
    yield assert_equal, mnc.get_data_dtype().type, np.uint8
    yield assert_equal, mnc.get_data_shape(), (91, 109, 91)
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
    # Check highest level load of minc works
    img = load(mnc_fname)
    data = img.get_data()
    yield assert_equal, data.shape, (91,109, 91)
    yield assert_equal, data.min(), 0.0
    yield assert_equal, data.max(), 1.0
    yield np.testing.assert_array_almost_equal, data.mean(), 0.27396803, 8
    # check dict-like stuff
    keys = mnc.keys()
    values = mnc.values()
    for i, key in enumerate(keys):
        yield assert_equal, values[i], mnc[key]
    items = mnc.items()
    # check if mnc can be converted to nifti
    ni_img = Nifti1Image.from_image(img)
    yield assert_array_equal, ni_img.get_affine(), aff
    yield assert_array_equal, ni_img.get_data(), data
