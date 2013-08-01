# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Tests for nifti2 reading package '''
from __future__ import division, print_function, absolute_import
import os

import numpy as np

from .. import nifti2
from ..nifti2 import (Nifti2Header, Nifti2PairHeader, Nifti2Image, Nifti2Pair)

from .test_nifti1 import (TestNifti1PairHeader, TestNifti1SingleHeader,
                          TestNifti1Pair, TestNifti1Image, TestNifti1General)

from ..testing import data_path

header_file = os.path.join(data_path, 'nifti2.hdr')
image_file = os.path.join(data_path, 'example4d2.nii.gz')


class _Nifti2Mixin(object):
    example_file = header_file
    sizeof_hdr = Nifti2Header.sizeof_hdr
    quat_dtype = np.float64

    def test_freesurfer_hack(self):
        # Disable this check
        pass


class TestNifti2PairHeader(_Nifti2Mixin, TestNifti1PairHeader):
    header_class = Nifti2PairHeader


class TestNifti2SingleHeader(_Nifti2Mixin, TestNifti1SingleHeader):
    header_class = Nifti2Header
    example_file = header_file


class TestNifti2Image(TestNifti1Image):
    # Run analyze-flavor spatialimage tests
    image_class = Nifti2Image


class TestNifti2Image(TestNifti1Image):
    # Run analyze-flavor spatialimage tests
    image_class = Nifti2Image


class TestNifti2Pair(TestNifti1Pair):
    # Run analyze-flavor spatialimage tests
    image_class = Nifti2Pair


class TestNifti2General(TestNifti1General):
    """ Test class to test nifti2 in general

    Tests here which mix the pair and the single type, and that should only be
    run once (not for each type) because they are slow
    """
    single_class = Nifti2Image
    pair_class = Nifti2Pair
    module = nifti2
    example_file = image_file
