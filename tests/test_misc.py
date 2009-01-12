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
from nifti import *
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


def suite():
    return unittest.makeSuite(MiscTests)


if __name__ == '__main__':
    unittest.main()

