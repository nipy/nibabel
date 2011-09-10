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

from ..testing import assert_equal, assert_raises

from . import test_spm99analyze


class TestSpm2AnalyzeHeader(test_spm99analyze.TestSpm99AnalyzeHeader):
    header_class = Spm2AnalyzeHeader

    def test_spm_scale_checks(self):
        # checks for scale
        hdr = self.header_class()
        hdr['scl_slope'] = np.nan
        fhdr, message, raiser = self.log_chk(hdr, 30)
        yield assert_equal(fhdr['scl_slope'], 1)
        problem_msg = ('no valid scaling in scalefactor '
                       '(=None) or cal / gl fields; '
                       'scalefactor assumed 1.0')
        yield assert_equal(message,
                           problem_msg +
                           '; setting scalefactor "scl_slope" to 1')
        yield assert_raises(*raiser)
        dxer = self.header_class.diagnose_binaryblock
        yield assert_equal(dxer(hdr.binaryblock),
                           problem_msg)
        hdr['scl_slope'] = np.inf
        yield assert_equal(dxer(hdr.binaryblock),
                           problem_msg)


class TestSpm2AnalyzeImage(test_spm99analyze.TestSpm99AnalyzeImage):
    # class for testing images
    image_class = Spm2AnalyzeImage


def test_origin_affine():
    # check that origin affine works, only
    hdr = Spm2AnalyzeHeader()
    aff = hdr.get_origin_affine()


def test_slope_inter():
    hdr = Spm2AnalyzeHeader()
    assert_equal(hdr.get_slope_inter(), (1.0, 0.0))
    for intup, outup in (((2.0,), (2.0, 0.0)),
                         ((None,), (None, None)),
                         ((1.0, None), (1.0, 0.0)),
                         ((0.0, None), (None, None)),
                         ((None, 0.0), (None, None))):
        hdr.set_slope_inter(*intup)
        assert_equal(hdr.get_slope_inter(), outup)
        # Check set survives through checking
        hdr = Spm2AnalyzeHeader.from_header(hdr, check=True)
        assert_equal(hdr.get_slope_inter(), outup)
    # Setting not-zero to offset raises error
    assert_raises(HeaderTypeError, hdr.set_slope_inter, None, 1.1)
    assert_raises(HeaderTypeError, hdr.set_slope_inter, 2.0, 1.1)

