# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Tests for SPM2 header stuff '''

import numpy as np

from ..spatialimages import HeaderTypeError
from ..spm2analyze import Spm2AnalyzeHeader, Spm2AnalyzeImage

from numpy.testing import assert_array_equal

from ..testing import assert_equal, assert_raises

from . import test_spm99analyze


class TestSpm2AnalyzeHeader(test_spm99analyze.TestSpm99AnalyzeHeader):
    header_class = Spm2AnalyzeHeader

    def test_slope_inter(self):
        hdr = self.header_class()
        assert_equal(hdr.get_slope_inter(), (1.0, 0.0))
        for intup, outup in (((2.0,), (2.0, 0.0)),
                            ((None,), (None, None)),
                            ((1.0, None), (1.0, 0.0)),
                            ((0.0, None), (None, None)), # Null scalings
                            ((np.nan, np.nan), (None, None)),
                            ((np.nan, None), (None, None)),
                            ((np.nan, None), (None, None)),
                            ((np.inf, None), (None, None)),
                            ((-np.inf, None), (None, None)),
                            ((None, 0.0), (None, None))):
            hdr.set_slope_inter(*intup)
            assert_equal(hdr.get_slope_inter(), outup)
            # Check set survives through checking
            hdr = Spm2AnalyzeHeader.from_header(hdr, check=True)
            assert_equal(hdr.get_slope_inter(), outup)
        # Setting not-zero to offset raises error
        assert_raises(HeaderTypeError, hdr.set_slope_inter, None, 1.1)
        assert_raises(HeaderTypeError, hdr.set_slope_inter, 2.0, 1.1)
        # Default slope is NaN
        hdr.set_slope_inter(None, None)
        assert_array_equal(hdr['scl_slope'], np.nan)


class TestSpm2AnalyzeImage(test_spm99analyze.TestSpm99AnalyzeImage):
    # class for testing images
    image_class = Spm2AnalyzeImage


def test_origin_affine():
    # check that origin affine works, only
    hdr = Spm2AnalyzeHeader()
    aff = hdr.get_origin_affine()
