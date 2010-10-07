# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
import os
from os.path import join as pjoin

import numpy as np
import numpy.testing.decorators as dec

from ..externals.netcdf import netcdf_file as netcdf

from .. import load, Nifti1Image
from ..minc import MincImage, MincFile
from ..spatialimages import ImageFileError
from nose.tools import assert_true, assert_equal, assert_false, \
    assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal
from ..testing import parametric, data_path

mnc_fname = os.path.join(data_path, 'tiny.mnc')
mnc_shape = (10,20,20)
mnc_affine = np.array([[0, 0, 2.0, -20],
                       [0, 2.0, 0, -20],
                       [2.0, 0, 0, -10],
                       [0, 0, 0, 1]])


@parametric
def test_mincfile():
    mnc = MincFile(netcdf(mnc_fname, 'r'))
    yield assert_equal(mnc.get_data_dtype().type, np.uint8)
    yield assert_equal(mnc.get_data_shape(), mnc_shape)
    yield assert_equal(mnc.get_zooms(), (2.0, 2.0, 2.0))
    yield assert_array_equal(mnc.get_affine(), mnc_affine)
    data = mnc.get_scaled_data()
    yield assert_equal(data.shape, mnc_shape)


@parametric
def test_load():
    # Check highest level load of minc works
    img = load(mnc_fname)
    data = img.get_data()
    yield assert_equal(data.shape, mnc_shape)
    # min, max, mean values from read in SPM2
    yield assert_array_almost_equal(data.min(), 0.20784314)
    yield assert_array_almost_equal(data.max(), 0.74901961)
    yield np.testing.assert_array_almost_equal(data.mean(), 0.60602819)
    # check if mnc can be converted to nifti
    ni_img = Nifti1Image.from_image(img)
    yield assert_array_equal(ni_img.get_affine(), mnc_affine)
    yield assert_array_equal(ni_img.get_data(), data)


@parametric
def test_compressed():
    # we can't read minc compreesed, raise error
    yield assert_raises(ImageFileError, load, 'test.mnc.gz')
    yield assert_raises(ImageFileError, load, 'test.mnc.bz2')
