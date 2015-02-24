# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Test for image funcs '''
from __future__ import division, print_function, absolute_import

import numpy as np

from ..funcs import concat_images, as_closest_canonical, OrientationError
from ..nifti1 import Nifti1Image
from ..loadsave import save

from ..tmpdirs import InTemporaryDirectory

from numpy.testing import assert_array_equal
from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)

_counter = 0
def _as_fname(img):
    global _counter
    fname = 'img%3d.nii' % _counter
    _counter = _counter + 1
    save(img, fname)
    return fname


def test_concat():
    shape = (1,2,5)
    data0 = np.arange(10).reshape(shape)
    affine = np.eye(4)
    img0_mem = Nifti1Image(data0, affine)
    data1 = data0 - 10
    img1_mem = Nifti1Image(data1, affine)
    img2_mem = Nifti1Image(data1, affine+1)
    img3_mem = Nifti1Image(data1.T, affine)
    all_data = np.concatenate(
        [data0[:,:,:,np.newaxis],data1[:,:,:,np.newaxis]],3)
    # Check filenames and in-memory images work
    with InTemporaryDirectory():
        imgs = [img0_mem, img1_mem, img2_mem, img3_mem]
        img_files = [_as_fname(img) for img in imgs]
        for img0, img1, img2, img3 in (imgs, img_files):
            all_imgs = concat_images([img0, img1])
            assert_array_equal(all_imgs.get_data(), all_data)
            assert_array_equal(all_imgs.affine, affine)
            # check that not-matching affines raise error
            assert_raises(ValueError, concat_images, [img0, img2])
            assert_raises(ValueError, concat_images, [img0, img3])
            # except if check_affines is False
            all_imgs = concat_images([img0, img1])
            assert_array_equal(all_imgs.get_data(), all_data)
            assert_array_equal(all_imgs.affine, affine)
        # Delete images as prophylaxis for windows access errors
        for img in imgs:
            del(img)

    # Test axis parameter and trailing unary dimension
    shape_4D = np.asarray(shape + (1,))
    data0 = np.arange(10).reshape(shape_4D)
    affine = np.eye(4)
    img0_mem = Nifti1Image(data0, affine)
    img1_mem = Nifti1Image(data0 - 10, affine)

    # 4d, same shape, append on axis 3
    concat_img1 = concat_images([img0_mem, img1_mem], axis=3)
    expected_shape1 = shape_4D.copy()
    expected_shape1[-1] *= 2
    assert_array_equal(concat_img1.shape, expected_shape1)

    # 4d, same shape, append on axis 0
    concat_img2 = concat_images([img0_mem, img1_mem], axis=0)
    expected_shape2 = shape_4D.copy()
    expected_shape2[0] *= 2
    assert_array_equal(concat_img2.shape, expected_shape2)

    # 4d, same shape, append on axis -1
    concat_img3 = concat_images([img0_mem, img1_mem], axis=-1)
    expected_shape3 = shape_4D.copy()
    expected_shape3[-1] *= 2
    assert_array_equal(concat_img3.shape, expected_shape3)

    # 4d, different shape, append on axis that's different
    print('%s %s' % (str(concat_img3.shape), str(img1_mem.shape)))
    concat_img4 = concat_images([concat_img3, img1_mem], axis=-1)
    expected_shape4 = shape_4D.copy()
    expected_shape4[-1] *= 3
    assert_array_equal(concat_img4.shape, expected_shape4)

    # 4d, different shape, append on axis that's not different...
    # Doesn't work!
    assert_raises(ValueError, concat_images, [concat_img3, img1_mem], axis=1)


def test_closest_canonical():
    arr = np.arange(24).reshape((2,3,4,1))
    # no funky stuff, returns same thing
    img = Nifti1Image(arr, np.eye(4))
    xyz_img = as_closest_canonical(img)
    assert_true(img is xyz_img)
    # a axis flip
    img = Nifti1Image(arr, np.diag([-1,1,1,1]))
    xyz_img = as_closest_canonical(img)
    assert_false(img is xyz_img)
    out_arr = xyz_img.get_data()
    assert_array_equal(out_arr, np.flipud(arr))
    # no error for enforce_diag in this case
    xyz_img = as_closest_canonical(img, True)
    # but there is if the affine is not diagonal
    aff = np.eye(4)
    aff[0,1] = 0.1
    # although it's more or less canonical already
    img = Nifti1Image(arr, aff)
    xyz_img = as_closest_canonical(img)
    assert_true(img is xyz_img)
    # it's still not diagnonal
    assert_raises(OrientationError, as_closest_canonical, img, True)
