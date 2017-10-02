# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
from __future__ import division, print_function, absolute_import

from os.path import join as pjoin

import numpy as np

from .. import load, Nifti1Image
from .. import brikhead

from nose.tools import (assert_true, assert_equal)
from numpy.testing import assert_array_equal
from ..testing import data_path

from .test_fileslice import slicer_samples
from .test_helpers import assert_data_similar

EXAMPLE_IMAGES = [
    dict(
        fname=pjoin(data_path, 'example4d+orig.BRIK'),
        shape=(33, 41, 25, 3),
        dtype=np.int16,
        affine=np.array([[-3.0,0,0,49.5],
                         [0,-3.0,0,82.312],
                         [0,0,3.0,-52.3511],
                         [0,0,0,1.0]]),
        zooms=(3., 3., 3., 3.),
        # These values from SPM2
        data_summary=dict(
            min=0,
            max=13722,
            mean=4266.76024636),
        is_proxy=True)
]


class TestAFNIImage(object):
    module = brikhead
    test_files = EXAMPLE_IMAGES

    def test_brikheadfile(self):
        for tp in self.test_files:
            brik = self.module.load(tp['fname'])
            assert_equal(brik.get_data_dtype().type, tp['dtype'])
            assert_equal(brik.shape, tp['shape'])
            assert_equal(brik.header.get_zooms(), tp['zooms'])
            assert_array_equal(brik.affine, tp['affine'])
            data = brik.get_data()
            assert_equal(data.shape, tp['shape'])

    def test_load(self):
        # Check highest level load of brikhead works
        for tp in self.test_files:
            img = load(tp['fname'])
            data = img.get_data()
            assert_equal(data.shape, tp['shape'])
            # min, max, mean values
            assert_data_similar(data, tp)
            # check if file can be converted to nifti
            ni_img = Nifti1Image.from_image(img)
            assert_array_equal(ni_img.affine, tp['affine'])
            assert_array_equal(ni_img.get_data(), data)

    def test_array_proxy_slicing(self):
        # Test slicing of array proxy
        for tp in self.test_files:
            img = load(tp['fname'])
            arr = img.get_data()
            prox = img.dataobj
            assert_true(prox.is_proxy)
            for sliceobj in slicer_samples(img.shape):
                assert_array_equal(arr[sliceobj], prox[sliceobj])
