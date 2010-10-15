# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Test for image funcs '''

from StringIO import StringIO

import numpy as np

import nibabel as nf

from ..funcs import concat_images, as_closest_canonical, OrientationError
from ..nifti1 import Nifti1Image

from numpy.testing import assert_array_equal
from nose.tools import (assert_true, assert_false,
                        assert_equal, assert_raises)
from ..testing import parametric


@parametric
def test_concat():
    shape = (1,2,5)
    data0 = np.arange(10).reshape(shape)
    affine = np.eye(4)
    img0 = nf.Nifti1Image(data0, affine)
    data1 = data0 - 10
    img1 = nf.Nifti1Image(data1, affine)
    all_imgs = concat_images([img0, img1])
    all_data = np.concatenate(
        [data0[:,:,:,np.newaxis],data1[:,:,:,np.newaxis]],3)
    yield assert_array_equal(all_imgs.get_data(), all_data)
    yield assert_array_equal(all_imgs.get_affine(), affine)
    # check that not-matching affines raise error
    img2 = nf.Nifti1Image(data1, affine+1)
    yield assert_raises(ValueError, concat_images, [img0, img2])
    img2 = nf.Nifti1Image(data1.T, affine)
    yield assert_raises(ValueError, concat_images, [img0, img2])
    # except if check_affines is False
    all_imgs = concat_images([img0, img1])
    yield assert_array_equal(all_imgs.get_data(), all_data)
    yield assert_array_equal(all_imgs.get_affine(), affine)
    

@parametric
def test_closest_canonical():
    arr = np.arange(24).reshape((2,3,4,1))
    # no funky stuff, returns same thing
    img = Nifti1Image(arr, np.eye(4))
    xyz_img = as_closest_canonical(img)
    yield assert_true(img is xyz_img)
    # a axis flip
    img = Nifti1Image(arr, np.diag([-1,1,1,1]))
    xyz_img = as_closest_canonical(img)
    yield assert_false(img is xyz_img)
    out_arr = xyz_img.get_data()
    yield assert_array_equal(out_arr, np.flipud(arr))
    # no error for enforce_diag in this case
    xyz_img = as_closest_canonical(img, True)
    # but there is if the affine is not diagonal
    aff = np.eye(4)
    aff[0,1] = 0.1
    # although it's more or less canonical already
    img = Nifti1Image(arr, aff)
    xyz_img = as_closest_canonical(img)
    yield assert_true(img is xyz_img)
    # it's still not diagnonal
    yield assert_raises(OrientationError, as_closest_canonical, img, True)
