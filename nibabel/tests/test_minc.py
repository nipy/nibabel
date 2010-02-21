import os
from os.path import join as pjoin

import numpy as np
import numpy.testing.decorators as dec

from nibabel.externals.netcdf import netcdf_file as netcdf

from nibabel import load, MincHeader, Nifti1Image
from nibabel.spatialimages import ImageFileError
from nose.tools import assert_true, assert_equal, assert_false, \
    assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nibabel.testing import parametric, data_path

mnc_fname = os.path.join(data_path, 'tiny.mnc')
   

@parametric
def test_eg_img():
    mnc = MincHeader(netcdf(mnc_fname, 'r'))
    img_shape = (10,20,20)
    yield assert_equal(mnc.get_data_dtype().type, np.uint8)
    yield assert_equal(mnc.get_data_shape(), img_shape)
    yield assert_equal(mnc.get_zooms(), (2.0, 2.0, 2.0))
    aff = np.array([[0, 0, 2.0, -20],
                    [0, 2.0, 0, -20],
                    [2.0, 0, 0, -10],
                    [0, 0, 0, 1]])
    yield assert_array_equal(mnc.get_best_affine(), aff)
    data = mnc.get_unscaled_data()
    yield assert_equal(data.shape, img_shape)
    data = mnc.get_scaled_data()
    yield assert_equal(data.shape, img_shape)
    # Check highest level load of minc works
    img = load(mnc_fname)
    data = img.get_data()
    yield assert_equal(data.shape, img_shape)
    # min, max, mean values from read in SPM2
    yield assert_array_almost_equal(data.min(), 0.20784314)
    yield assert_array_almost_equal(data.max(), 0.74901961)
    yield np.testing.assert_array_almost_equal(data.mean(), 0.60602819)
    # check dict-like stuff
    keys = mnc.keys()
    values = mnc.values()
    for i, key in enumerate(keys):
        yield assert_equal(values[i], mnc[key])
    items = mnc.items()
    # check if mnc can be converted to nifti
    ni_img = Nifti1Image.from_image(img)
    yield assert_array_equal(ni_img.get_affine(), aff)
    yield assert_array_equal(ni_img.get_data(), data)


@parametric
def test_compressed():
    # we can't read minc compreesed, raise error
    yield assert_raises(ImageFileError, load, 'test.mnc.gz')
    yield assert_raises(ImageFileError, load, 'test.mnc.bz2')
