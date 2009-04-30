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

import os

import numpy as N

from cnifti.format import NiftiFormat

from nifti.testing import example_data_path

import unittest


class NiftiFormatTests(unittest.TestCase):

    def testLoadHdrFromFile(self):
        nhdr = NiftiFormat(os.path.join(example_data_path, 'example4d.nii.gz'))

        # basic incomplete property check
        self.failUnlessEqual(nhdr.extent, (128, 96, 24, 2))
        self.failUnlessEqual(nhdr.rtime, 2000)
        self.failUnlessEqual(nhdr.pixdim,
                             (2.0, 2.0, 2.1999990940093994, 2000.0,
                              1.0, 1.0, 1.0))
        self.failUnlessEqual(nhdr.extent, (128, 96, 24, 2))
        self.failUnlessEqual(nhdr.voxdim,
                             (2.0, 2.0, 2.1999990940093994))
        self.failUnlessEqual(nhdr.extent, (128, 96, 24, 2))
        self.failUnlessEqual(nhdr.header['datatype'], 4)
        self.failUnlessEqual(nhdr.header['bitpix'], 16)
        self.failUnlessEqual(nhdr.header['magic'], 'n+1')


    def testLoadHdrFromArray(self):
        nhdr = NiftiFormat(N.zeros((4,3,2)))

        # basic incomplete property check
        self.failUnlessEqual(nhdr.extent, (2, 3, 4))
        self.failUnlessEqual(nhdr.rtime, 0)



def suite():
    return unittest.makeSuite(NiftiFormatTests)


if __name__ == '__main__':
    unittest.main()

