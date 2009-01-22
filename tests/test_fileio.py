#vim:fileencoding=utf-8
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyNIfTI file io"""

__docformat__ = 'restructuredtext'

from nifti.image import NiftiImage, MemMappedNiftiImage
from nifti.format import NiftiFormat
import nifti.clib as ncl
import unittest
import md5
import tempfile
import shutil
import os
import numpy as N


def md5sum(filename):
    """ Generate MD5 hash string.
    """
    file = open( filename )
    sum = md5.new()
    while True:
        data = file.read()
        if not data:
            break
        sum.update(data)
    return sum.hexdigest()


class FileIOTests(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp('pynifti_test')
        self.nimg = NiftiImage(os.path.join('data', 'example4d.nii.gz'))
        self.fp = tempfile.NamedTemporaryFile(suffix='.nii.gz')


    def tearDown(self):
        shutil.rmtree(self.workdir)
        del self.nimg
        self.fp.close()


    def testIdempotentLoadSaveCycle(self):
        """ check if file is unchanged by load/save cycle.
        """
        md5_orig = md5sum(os.path.join('data', 'example4d.nii.gz'))
        self.nimg.save( os.path.join( self.workdir, 'iotest.nii.gz') )
        nimg2 = NiftiImage( os.path.join( self.workdir, 'iotest.nii.gz'))
        md5_io =  md5sum( os.path.join( self.workdir, 'iotest.nii.gz') )

        self.failUnlessEqual(md5_orig, md5_io)


    def testUnicodeLoadSaveCycle(self):
        """ check load/save cycle for unicode filenames.
        """
        md5_orig = md5sum(os.path.join('data', 'example4d.nii.gz'))
        self.nimg.save( os.path.join( self.workdir, 'üöä.nii.gz') )
        md5_io =  md5sum( os.path.join( self.workdir, 'üöä.nii.gz') )

        self.failUnlessEqual(md5_orig, md5_io)


    def testDataAccess(self):
        # test two points
        self.failUnlessEqual(self.nimg.data[1,12,59,49], 509)
        self.failUnlessEqual(self.nimg.data[0,4,17,42], 435)


    def testDataOwnership(self):
        # assign data, but no copying
        data = self.nimg.data
        # data is a view of the array data buffer
        assert data.flags.owndata == False

        # get copy
        data_copy = self.nimg.asarray()
        # data_copy is a copy of the array data buffer, it own's the buffer
        assert data_copy.flags.owndata == True

        # test two points
        self.failUnlessEqual(data[1,12,59,49], 509)
        self.failUnlessEqual(data[0,4,17,42], 435)

        # now remove image and try again
        #del nimg
        # next section would cause segfault as the 
        #self.failUnlessEqual(data[1,12,59,49], 509)
        #self.failUnlessEqual(data[0,4,17,42], 435)

        self.failUnlessEqual(data_copy[1,12,59,49], 509)
        self.failUnlessEqual(data_copy[0,4,17,42], 435)


    def testFromScratch(self):
        data = N.arange(24).reshape(2,3,4)
        n = NiftiImage(data)

        n.save(os.path.join(self.workdir, 'scratch.nii'))

        n2 = NiftiImage(os.path.join(self.workdir, 'scratch.nii'))

        self.failUnless((n2.data == data).all())

        # now modify data and store again
        n2.data[:] = n2.data * 2

        n2.save(os.path.join(self.workdir, 'scratch.nii'))

        # reopen and check data
        n3 = NiftiImage(os.path.join(self.workdir, 'scratch.nii'))

        self.failUnless((n3.data == data * 2).all())


#    def testLeak(self):
#        for i in xrange(100000):
#            nimg = NiftiImage('data/example4d.nii.gz')
#            nimg = NiftiImage(N.arange(1))

    def testMemoryMapping(self):
        # save as uncompressed file
        self.nimg.save(os.path.join(self.workdir, 'mmap.nii'))

        nimg_mm = MemMappedNiftiImage(os.path.join(self.workdir, 'mmap.nii'))

        # make sure we have the same
        self.failUnlessEqual(self.nimg.data[1,12,39,46],
                             nimg_mm.data[1,12,39,46])

        orig = nimg_mm.data[0,12,30,23]
        nimg_mm.data[0,12,30,23] = 999

        # make sure data is written to disk
        nimg_mm.save()

        self.failUnlessEqual(nimg_mm.data[0,12,30,23], 999)

        # now reopen non-mapped and confirm operation
        nimg_mod = NiftiImage(os.path.join(self.workdir, 'mmap.nii'))
        self.failUnlessEqual(nimg_mod.data[0,12,30,23], 999)

        self.failUnlessRaises(RuntimeError, nimg_mm.setFilename, 'someother')


    def testQFormSetting(self):
        # 4x4 identity matrix
        ident = N.identity(4)
        self.failIf( (self.nimg.qform == ident).all() )

        # assign new qform
        self.nimg.qform = ident
        self.failUnless( (self.nimg.qform == ident).all() )

        # test save/load cycle
        self.nimg.save( os.path.join( self.workdir, 'qformtest.nii.gz') )
        nimg2 = NiftiImage( os.path.join( self.workdir,
                                               'qformtest.nii.gz') )

        self.failUnless( (self.nimg.qform == nimg2.qform).all() )


    def testQFormSetting_fromFile(self):
        # test setting qoffset
        new_qoffset = (10.0, 20.0, 30.0)
        self.nimg.qoffset = new_qoffset

        fname = os.path.join(self.workdir, 'test-qoffset-file.nii.gz')
        self.nimg.save(fname)
        nimg2 = NiftiImage(fname)

        self.failUnless((self.nimg.qform == nimg2.qform).all())

        # now test setting full qform
        nimg3 = NiftiImage(os.path.join('data', 'example4d.nii.gz'))

        # build custom qform matrix
        qform = N.identity(4)
        # give 2mm pixdim
        qform.ravel()[0:11:5] = 2
        # give some qoffset
        qform[0:3, 3] = [10.0, 20.0, 30.0]

        nimg3.qform = qform
        self.failUnless( (nimg3.qform == qform).all() )

        # see whether it survives save/load cycle
        fname = os.path.join(self.workdir, 'qform_fromfile_test.nii.gz')
        nimg3.save(fname)
        nimg4 = NiftiImage(fname)

        # test rotation portion of qform, pixdims appear to be ok
        self.failUnless( (nimg4.qform[:3, :3] == qform[:3, :3]).all() )
        # test full qform
        self.failUnless( (nimg4.qform == qform).all() )


    def testQFormSetting_fromArray(self):
        data = N.zeros((4,3,2))
        ident = N.identity(4)
        # give 2mm pixdim
        ident.ravel()[0:11:5] = 2
        # give some qoffset
        ident[0:3, 3] = [10.0, 20.0, 30.0]
        nimg = NiftiImage(data)
        nimg.qform = ident
        self.failUnless( (nimg.qform == ident).all() )

        fname = os.path.join(self.workdir, 'qform_fromarray_test.nii.gz')
        nimg.save(fname)
        nimg2 = NiftiImage(fname)
        # test rotation portion of qform, pixdims appear to be ok
        self.failUnless( (nimg.qform[:3, :3] == nimg2.qform[:3, :3]).all() )
        # test full qform
        self.failUnless( (nimg.qform == nimg2.qform).all() )


    def test_setPixDims(self):
        pdims = (20.0, 30.0, 40.0, 2000.0, 1.0, 1.0, 1.0)
        # test setPixDims func
        self.nimg.setPixDims(pdims)
        self.failUnless(self.nimg.getPixDims() == pdims)

        self.nimg.save(self.fp.name)
        nimg2 = NiftiImage(self.fp.name)
        self.failUnless(nimg2.getPixDims() == pdims)

        # test assignment
        pdims = [x*2 for x in pdims]
        pdims = tuple(pdims)
        self.nimg.pixdim = pdims
        self.failUnless(self.nimg.pixdim == pdims)

        self.nimg.save(self.fp.name)
        nimg2 = NiftiImage(self.fp.name)
        self.failUnless(nimg2.pixdim == pdims)


    def test_setVoxDims(self):
        vdims = (2.0, 3.0, 4.0)
        # test setVoxDims func
        self.nimg.setVoxDims(vdims)
        self.failUnless(self.nimg.getVoxDims() == vdims)

        self.nimg.save(self.fp.name)
        nimg2 = NiftiImage(self.fp.name)
        self.failUnless(nimg2.getVoxDims() == vdims)

        # test assignment
        vdims = [x*2 for x in vdims]
        vdims = tuple(vdims)
        self.nimg.voxdim = vdims
        self.failUnless(self.nimg.voxdim == vdims)

        self.nimg.save(self.fp.name)
        nimg2 = NiftiImage(self.fp.name)
        self.failUnless(self.nimg.voxdim == nimg2.voxdim)


    def testExtensionsSurvive(self):
        """ check if extensions actually get safed to the file.
        """
        self.nimg.extensions += ('comment', 'fileio')
        self.nimg.save(os.path.join( self.workdir, 'extensions.nii.gz'))

        nimg2 = NiftiImage(os.path.join(self.workdir, 'extensions.nii.gz'))

        # should be the last one added
        self.failUnless(nimg2.extensions[-1] == 'fileio')


    def testMetaReconstruction(self):
        """Check if the meta data gets properly reconstructed during
        save/load cycle.
        """
        self.nimg.meta['something'] = 'Gmork'
        self.nimg.save(os.path.join( self.workdir, 'meta.nii.gz'))
        nimg2 = NiftiImage(os.path.join(self.workdir, 'meta.nii.gz'))

        self.failUnless(nimg2.meta['something'] == 'Gmork')


    def testArrayAssign(self):
        alt_array = N.zeros((3,4,5,6,7), dtype='int')
        self.nimg.data = alt_array

        self.nimg.save(os.path.join( self.workdir, 'assign.nii'))
        nimg2 = NiftiImage(os.path.join(self.workdir, 'assign.nii'))


        self.failUnless(nimg2.header['dim'] == [5, 7, 6, 5, 4, 3, 1, 1])
        self.failUnless(nimg2.raw_nimg.datatype == ncl.NIFTI_TYPE_INT32)


def suite():
    return unittest.makeSuite(FileIOTests)


if __name__ == '__main__':
    unittest.main()

