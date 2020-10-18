# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Tests for image statistics """

import numpy as np

from .. import imagestats
from nibabel.testing import test_data
from nibabel.loadsave import load

import pytest


def test_mask_volume():
    # Test mask volume computation
    infile = test_data(fname="anatomical.nii")
    img = load(infile)
    vol_mm3 = imagestats.mask_volume(img)
    vol_vox = imagestats.mask_volume(img, units='vox')

    assert float(vol_mm3) == 2273328656.0
    assert float(vol_vox) == 284166082.0

