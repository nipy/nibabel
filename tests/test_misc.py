#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Misc unit tests for PyNIfTI"""

__docformat__ = 'restructuredtext'

import os

import numpy as N

from nifti import *
import nifti.clib as ncl
from nifti.format import NiftiFormat
import unittest


class MiscTests(unittest.TestCase):
    def testFilenameProps(self):
        def helper(obj, filename):
            obj.filename = filename

        nif = NiftiFormat(os.path.join('data', 'example4d'))
        self.failUnlessRaises(AttributeError, helper, nif, 'test.nii')

        nim = NiftiImage(os.path.join('data', 'example4d'))
        nim.filename = 'test.nii'
        self.failUnless(nim.filename == 'test.nii')


    def testArrayAssign(self):
        """Test whether the header is updated correctly when assigning a new
        data array"""
        orig_array = N.ones((2,3,4), dtype='float')
        nimg = NiftiImage(orig_array)

        self.failUnless(nimg.header['dim'] == [3, 4, 3, 2, 1, 1, 1, 1])
        self.failUnless(nimg.raw_nimg.datatype == ncl.NIFTI_TYPE_FLOAT64)

        # now turn that image into 4d with ints
        alt_array = N.zeros((4,5,6,7), dtype='int')
        nimg.data = alt_array

        self.failUnless(nimg.data.shape == alt_array.shape)
        self.failUnless(nimg.header['dim'] == [4, 7, 6, 5, 4, 1, 1, 1])
        self.failUnless(nimg.raw_nimg.datatype == ncl.NIFTI_TYPE_INT32)


def suite():
    return unittest.makeSuite(MiscTests)


if __name__ == '__main__':
    unittest.main()

