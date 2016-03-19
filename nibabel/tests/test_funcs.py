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
    # Smoke test: concat empty list.
    assert_raises(ValueError, concat_images, [])

    # Build combinations of 3D, 4D w/size[3] == 1, and 4D w/size[3] == 3
    all_shapes_5D = ((1, 4, 5, 3, 3),
                     (7, 3, 1, 4, 5),
                     (0, 2, 1, 4, 5))

    affine = np.eye(4)
    for dim in range(2, 6):
        all_shapes_ND = tuple((shape[:dim] for shape in all_shapes_5D))
        all_shapes_N1D_unary = tuple((shape + (1,) for shape in all_shapes_ND))
        all_shapes = all_shapes_ND + all_shapes_N1D_unary

        # Loop over all possible combinations of images, in first and
        #   second position.
        for data0_shape in all_shapes:
            data0_numel = np.asarray(data0_shape).prod()
            data0 = np.arange(data0_numel).reshape(data0_shape)
            img0_mem = Nifti1Image(data0, affine)

            for data1_shape in all_shapes:
                data1_numel = np.asarray(data1_shape).prod()
                data1 = np.arange(data1_numel).reshape(data1_shape)
                img1_mem = Nifti1Image(data1, affine)
                img2_mem = Nifti1Image(data1, affine + 1)  # bad affine

                # Loop over every possible axis, including None (explicit and implied)
                for axis in (list(range(-(dim - 2), (dim - 1))) + [None, '__default__']):

                    # Allow testing default vs. passing explicit param
                    if axis == '__default__':
                        np_concat_kwargs = dict(axis=-1)
                        concat_imgs_kwargs = dict()
                        axis = None  # Convert downstream
                    elif axis is None:
                        np_concat_kwargs = dict(axis=-1)
                        concat_imgs_kwargs = dict(axis=axis)
                    else:
                        np_concat_kwargs = dict(axis=axis)
                        concat_imgs_kwargs = dict(axis=axis)

                    # Create expected output
                    try:
                        # Error will be thrown if the np.concatenate fails.
                        #   However, when axis=None, the concatenate is possible
                        #   but our efficient logic (where all images are
                        #   3D and the same size) fails, so we also
                        #   have to expect errors for those.
                        if axis is None:  # 3D from here and below
                            all_data = np.concatenate([data0[..., np.newaxis],
                                                       data1[..., np.newaxis]],
                                                      **np_concat_kwargs)
                        else:  # both 3D, appending on final axis
                            all_data = np.concatenate([data0, data1],
                                                      **np_concat_kwargs)
                        expect_error = False
                    except ValueError:
                        # Shapes are not combinable
                        expect_error = True

                    # Check filenames and in-memory images work
                    with InTemporaryDirectory():
                        # Try mem-based, file-based, and mixed
                        imgs = [img0_mem, img1_mem, img2_mem]
                        img_files = [_as_fname(img) for img in imgs]
                        imgs_mixed = [imgs[0], img_files[1], imgs[2]]
                        for img0, img1, img2 in (imgs, img_files, imgs_mixed):
                            try:
                                all_imgs = concat_images([img0, img1],
                                                         **concat_imgs_kwargs)
                            except ValueError as ve:
                                assert_true(expect_error, str(ve))
                            else:
                                assert_false(
                                    expect_error, "Expected a concatenation error, but got none.")
                                assert_array_equal(all_imgs.get_data(), all_data)
                                assert_array_equal(all_imgs.affine, affine)

                            # check that not-matching affines raise error
                            assert_raises(ValueError, concat_images, [
                                          img0, img2], **concat_imgs_kwargs)

                            # except if check_affines is False
                            try:
                                all_imgs = concat_images([img0, img1], **concat_imgs_kwargs)
                            except ValueError as ve:
                                assert_true(expect_error, str(ve))
                            else:
                                assert_false(
                                    expect_error, "Expected a concatenation error, but got none.")
                                assert_array_equal(all_imgs.get_data(), all_data)
                                assert_array_equal(all_imgs.affine, affine)


def test_closest_canonical():
    arr = np.arange(24).reshape((2, 3, 4, 1))
    # no funky stuff, returns same thing
    img = Nifti1Image(arr, np.eye(4))
    xyz_img = as_closest_canonical(img)
    assert_true(img is xyz_img)
    # a axis flip
    img = Nifti1Image(arr, np.diag([-1, 1, 1, 1]))
    xyz_img = as_closest_canonical(img)
    assert_false(img is xyz_img)
    out_arr = xyz_img.get_data()
    assert_array_equal(out_arr, np.flipud(arr))
    # no error for enforce_diag in this case
    xyz_img = as_closest_canonical(img, True)
    # but there is if the affine is not diagonal
    aff = np.eye(4)
    aff[0, 1] = 0.1
    # although it's more or less canonical already
    img = Nifti1Image(arr, aff)
    xyz_img = as_closest_canonical(img)
    assert_true(img is xyz_img)
    # it's still not diagnonal
    assert_raises(OrientationError, as_closest_canonical, img, True)
