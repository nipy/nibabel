""" Testing imageclasses module
"""

from os.path import dirname, join as pjoin

import numpy as np

from nibabel.optpkg import optional_package

import nibabel as nib
from nibabel.analyze import AnalyzeImage
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image

from nibabel.imageclasses import spatial_axes_first

from nose.tools import (assert_true, assert_false)

DATA_DIR = pjoin(dirname(__file__), 'data')

have_h5py = optional_package('h5py')[1]

MINC_3DS = ('minc1_1_scale.mnc',)
MINC_4DS = ('minc1_4d.mnc',)
if have_h5py:
    MINC_3DS = MINC_3DS + ('minc2_1_scale.mnc',)
    MINC_4DS = MINC_4DS + ('minc2_4d.mnc',)


def test_spatial_axes_first():
    # Function tests is spatial axes are first three axes in image
    # Always True for Nifti and friends
    affine = np.eye(4)
    for shape in ((2, 3), (4, 3, 2), (5, 4, 1, 2), (2, 3, 5, 2, 1)):
        for img_class in (AnalyzeImage, Nifti1Image, Nifti2Image):
            data = np.zeros(shape)
            img = img_class(data, affine)
            assert_true(spatial_axes_first(img))
    # True for MINC images < 4D
    for fname in MINC_3DS:
        img = nib.load(pjoin(DATA_DIR, fname))
        assert_true(len(img.shape) == 3)
        assert_true(spatial_axes_first(img))
    # False for MINC images < 4D
    for fname in MINC_4DS:
        img = nib.load(pjoin(DATA_DIR, fname))
        assert_true(len(img.shape) == 4)
        assert_false(spatial_axes_first(img))
