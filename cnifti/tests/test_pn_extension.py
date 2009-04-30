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

import os
import unittest
import numpy as N

from cnifti.format import NiftiFormat
from cnifti import NiftiImage

from nifti.testing import example_data_path

class NiftiExtensionTests(unittest.TestCase):

    def testExtensions(self):
        nim = NiftiFormat(os.path.join(example_data_path, 'example4d.nii.gz'))
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


    def testMetaData(self):
        # come up with image
        nim = NiftiImage(N.arange(24).reshape(1,2,3,4))
        nim.meta['test1'] = range(5)

        # test whether meta data makes it into header dict
        self.failUnless(nim.header.has_key('meta'))
        self.failUnless(nim.header['meta']['test1'] == range(5))

        # clone image
        # test whether meta data makes it again into header dict
        nim2 = NiftiImage(nim.data, nim.header)
        self.failUnless(nim2.header.has_key('meta'))
        self.failUnless(nim2.header['meta']['test1'] == range(5))


def suite():
    return unittest.makeSuite(NiftiExtensionTests)


if __name__ == '__main__':
    unittest.main()

