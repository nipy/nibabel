""" Test we can correctly import example PARREC files
"""
from __future__ import print_function, absolute_import

from glob import glob
from os.path import join as pjoin, basename, splitext, exists

import numpy as np

from .. import load as top_load
from ..parrec import load

from .nibabel_data import get_nibabel_data, needs_nibabel_data

from nose import SkipTest
from nose.tools import assert_true, assert_false, assert_equal

from numpy.testing import assert_almost_equal

BALLS = pjoin(get_nibabel_data(), 'nitest-balls1')

# Amount by which affine translation differs from NIFTI conversion
AFF_OFF = [-0.93644031, -0.95572686, 0.03288748]


@needs_nibabel_data('nitest-balls1')
def test_loading():
    # Test loading of parrec files
    for par in glob(pjoin(BALLS, 'PARREC', '*.PAR')):
        par_root, ext = splitext(basename(par))
        # NA.PAR appears to be a localizer, with three slices in each of the
        # three orientations: sagittal; coronal, transverse
        if par_root ==  'NA':
            continue
        # Check we can load the image
        pimg = load(par)
        assert_equal(pimg.shape[:3], (80, 80, 10))
        # Compare against NIFTI if present
        nifti_fname = pjoin(BALLS, 'NIFTI', par_root + '.nii.gz')
        if exists(nifti_fname):
            nimg = top_load(nifti_fname)
            assert_almost_equal(nimg.affine[:3, :3], pimg.affine[:3, :3], 3)
            # The translation part is always off by the same ammout
            aff_off = pimg.affine[:3, 3] - nimg.affine[:3, 3]
            assert_almost_equal(aff_off, AFF_OFF, 4)
            # The difference is max in the order of 0.5 voxel
            vox_sizes = np.sqrt((nimg.affine[:3, :3] ** 2).sum(axis=0))
            assert_true(np.all(np.abs(aff_off / vox_sizes) <= 0.5))
            # The data is very close, unless it's the fieldmap
            if par_root != 'fieldmap':
                assert_true(np.allclose(pimg.dataobj, nimg.dataobj))
            # Not sure what's going on with the fieldmap image - TBA


@needs_nibabel_data('nitest-balls1')
def test_fieldmap():
    # Test fieldmap image
    # Exploring the DICOM suggests that the first volume is magnitude and the
    # second is phase.  The NIfTI has very odd scaling, being all negative.
    fieldmap_par = pjoin(BALLS, 'PARREC', 'fieldmap.PAR')
    fieldmap_nii = pjoin(BALLS, 'NIFTI', 'fieldmap.nii.gz')
    pimg = load(fieldmap_par)
    nimg = top_load(fieldmap_nii)
    raise SkipTest('Fieldmap remains puzzling')
