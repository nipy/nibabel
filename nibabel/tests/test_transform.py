# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests of the transform module."""
import os
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal, \
    assert_array_almost_equal
import pytest

from ..loadsave import load as loadimg
from ..nifti1 import Nifti1Image
from ..eulerangles import euler2mat
from ..affines import from_matvec
from ..volumeutils import shape_zoom_affine
from ..transform import linear as nbl
from ..testing import (assert_equal, assert_not_equal, assert_true,
                       assert_false, assert_raises, data_path,
                       suppress_warnings, assert_dt_equal)
from ..tmpdirs import InTemporaryDirectory


SOMEONES_ANATOMY = os.path.join(data_path, 'someones_anatomy.nii.gz')
# SOMEONES_ANATOMY = os.path.join(data_path, 'someones_anatomy.nii.gz')


@pytest.mark.parametrize('image_orientation', ['RAS', 'LAS', 'LPS', 'oblique'])
def test_affines_save(image_orientation):
    """Check implementation of exporting affines to formats."""
    # Generate test transform
    img = loadimg(SOMEONES_ANATOMY)
    imgaff = img.affine

    if image_orientation == 'LAS':
        newaff = imgaff.copy()
        newaff[0, 0] *= -1.0
        newaff[0, 3] = imgaff.dot(np.hstack((np.array(img.shape[:3]) - 1, 1.0)))[0]
        img = Nifti1Image(np.flip(img.get_fdata(), 0), newaff, img.header)
    elif image_orientation == 'LPS':
        newaff = imgaff.copy()
        newaff[0, 0] *= -1.0
        newaff[1, 1] *= -1.0
        newaff[:2, 3] = imgaff.dot(np.hstack((np.array(img.shape[:3]) - 1, 1.0)))[:2]
        img = Nifti1Image(np.flip(np.flip(img.get_fdata(), 0), 1), newaff, img.header)
    elif image_orientation == 'oblique':
        A = shape_zoom_affine(img.shape, img.header.get_zooms(), x_flip=False)
        R = from_matvec(euler2mat(x=0.09, y=0.001, z=0.001))
        newaff = R.dot(A)
        img = Nifti1Image(img.get_fdata(), newaff, img.header)
        img.header.set_qform(newaff, 1)
        img.header.set_sform(newaff, 1)

    T = from_matvec(euler2mat(x=0.9, y=0.001, z=0.001), [4.0, 2.0, -1.0])

    xfm = nbl.Affine(T)
    xfm.reference = img

    itk = nbl.load(os.path.join(data_path, 'affine-%s-itk.tfm' % image_orientation),
                   fmt='itk')
    fsl = np.loadtxt(os.path.join(data_path, 'affine-%s.fsl' % image_orientation))

    with InTemporaryDirectory():
        xfm.to_filename('M.tfm', fmt='itk')
        xfm.to_filename('M.fsl', fmt='fsl')

        nb_itk = nbl.load('M.tfm', fmt='itk')
        nb_fsl = np.loadtxt('M.fsl')

    assert_equal(itk, nb_itk)
    assert_almost_equal(fsl, nb_fsl)

# Create version not aligned to canonical
