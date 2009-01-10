#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyNIfTI extension handling"""

__docformat__ = 'restructuredtext'

from nifti.format import NiftiFormat
from nifti import NiftiImage

import os
import unittest
import numpy as N


class NiftiExtensionTests(unittest.TestCase):
    def testExtensions(self):
        nim = NiftiFormat(os.path.join('data', 'example4d.nii.gz'))
        # basic checks of the available extensions
        ext = nim.extensions
        self.failUnless(len(ext) == 2)
        self.failUnless(ext.count('comment') == 2)
        self.failUnless(ext.count('afni') == 0)
        self.failUnless(ext.ecodes == [6, 6])

        # first extension should be short one
        self.failUnless(ext[0] == 'extcomment1')

        # add one
        ext += ('afni', '<xml></xml>')
        self.failUnless(ext.ecodes == [6, 6, 4])
        self.failUnless(ext.count('comment') == 2)
        self.failUnless(ext.count('afni') == 1)


        # delete one
        del ext[1]
        self.failUnless(ext.ecodes == [6, 4])
        self.failUnless(ext.count('comment') == 1)
        self.failUnless(ext.count('afni') == 1)


def suite():
    return unittest.makeSuite(NiftiExtensionTests)


if __name__ == '__main__':
    unittest.main()

