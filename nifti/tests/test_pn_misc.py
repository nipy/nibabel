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
import cPickle


class MiscTests(unittest.TestCase):
    def setUp(self):
        data_path, _ = os.path.split(__file__)
        self.data_path = os.path.join(data_path, 'data')

    def testFilenameProps(self):
        def helper(obj, filename):
            obj.filename = filename

        nif = NiftiFormat(os.path.join(self.data_path, 'example4d'))
        self.failUnlessRaises(AttributeError, helper, nif, 'test.nii')

        nim = NiftiImage(os.path.join(self.data_path, 'example4d'))
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
        alt_array = N.zeros((4,5,6,7), dtype='int32')
        nimg.data = alt_array

        self.failUnless(nimg.data.shape == alt_array.shape)
        self.failUnless(nimg.header['dim'] == [4, 7, 6, 5, 4, 1, 1, 1])
        self.failUnless(nimg.raw_nimg.datatype == ncl.NIFTI_TYPE_INT32)


    def testCopying(self):
        nim = NiftiImage(os.path.join(self.data_path, 'example4d'))

        n2 = nim.copy()
        n2.voxdim = (2,3,4)
        n2.data[0,3,4,2] = 543

        self.failUnless(n2.voxdim == (2,3,4))
        self.failIf(nim.voxdim == n2.voxdim)

        self.failUnless(n2.data[0,3,4,2] == 543)
        self.failIf(nim.data[0,3,4,2] == n2.data[0,3,4,2])


# XXX Disabled since the corresponding method is temporally unavailable.
#
#    def testVolumeIter(self):
#        nim = NiftiImage(os.path.join('data', 'example4d'))
#
#        vols = [v for v in nim.iterVolumes()]
#
#        self.failUnless(len(vols) == 2)
#
#        for v in vols:
#            self.failUnless(v.extent == v.volextent == nim.volextent)
#
#        # test if data is shared
#        vols[1].data[20,10,5] = 666
#
#        # check if copying works
#        print vols[1].data[20,10,5]
#        print nim.data[1,20,10,5]


    def testPickleCycle(self):
        nim = NiftiImage(os.path.join(self.data_path, 'example4d'))

        pickled = cPickle.dumps(nim)

        nim2 = cPickle.loads(pickled)

        self.failUnless((nim.data == nim2.data).all())
        self.failUnless(N.all([N.all(nim2.header[k] == v)
                for k,v in nim.header.iteritems()]))


def suite():
    return unittest.makeSuite(MiscTests)


if __name__ == '__main__':
    unittest.main()

