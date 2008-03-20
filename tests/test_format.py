#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyNIfTI header class"""

__docformat__ = 'restructuredtext'

from nifti.niftiformat import NiftiFormat
import unittest
import numpy as N


class NiftiFormatTests(unittest.TestCase):
    def testLoadHdrFromFile(self):
        nhdr = NiftiFormat('data/example4d.nii.gz')

        # basic incomplete property check
        self.failUnlessEqual(nhdr.extent, (128, 96, 24, 2))
        self.failUnlessEqual(nhdr.rtime, 2000)


    def testLoadHdrFromArray(self):
        nhdr = NiftiFormat(N.zeros((4,3,2)))

        # basic incomplete property check
        self.failUnlessEqual(nhdr.extent, (2, 3, 4))
        self.failUnlessEqual(nhdr.rtime, 0)



def suite():
    return unittest.makeSuite(NiftiFormatTests)


if __name__ == '__main__':
    unittest.main()

