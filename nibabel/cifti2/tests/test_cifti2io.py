# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
from __future__ import division, print_function, absolute_import

from os.path import join as pjoin, dirname
import sys
from distutils.version import LooseVersion

import numpy as np

import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory

from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_raises)

IO_DATA_PATH = pjoin(dirname(__file__), 'data')
DATA_FILE1 = pjoin(IO_DATA_PATH, '')
DATA_FILE2 = pjoin(IO_DATA_PATH,
                   'Conte69.MyelinAndCorrThickness.32k_fs_LR.dscalar.nii')
DATA_FILE3 = pjoin(IO_DATA_PATH,
                   'Conte69.MyelinAndCorrThickness.32k_fs_LR.dtseries.nii')
DATA_FILE4 = pjoin(IO_DATA_PATH,
                   'Conte69.MyelinAndCorrThickness.32k_fs_LR.ptseries.nii')
DATA_FILE5 = pjoin(IO_DATA_PATH,
                   'Conte69.parcellations_VGD11b.32k_fs_LR.dlabel.nii')
DATA_FILE6 = pjoin(IO_DATA_PATH, 'ones.dscalar.nii')
datafiles = [DATA_FILE2, DATA_FILE3, DATA_FILE4, DATA_FILE5, DATA_FILE6]


def test_read_ordering():
    # DATA_FILE1 has an expected darray[0].data shape of (3,3).  However if we
    # read another image first (DATA_FILE2) then the shape is wrong
    # Read an image
    img2 = nib.load(DATA_FILE6)
    assert_equal(img2.data.shape, (91282,))


def test_version():
    for i, dat in enumerate(datafiles):
        img = nib.load(dat)
        assert_equal(LooseVersion(img.header.version), LooseVersion('2'))

'''
def test_dataarray1():
    img1 = gi.read(DATA_FILE1)
    # Round trip
    with InTemporaryDirectory():
        gi.write(img1, 'test.gii')
        bimg = gi.read('test.gii')
    for img in (img1, bimg):
        assert_array_almost_equal(img.darrays[0].data, DATA_FILE1_darr1)
        assert_array_almost_equal(img.darrays[1].data, DATA_FILE1_darr2)
        me=img.darrays[0].meta.get_metadata()
        assert_true('AnatomicalStructurePrimary' in me)
        assert_true('AnatomicalStructureSecondary' in me)
        assert_equal(me['AnatomicalStructurePrimary'], 'CortexLeft')
        assert_array_almost_equal(img.darrays[0].coordsys.xform, np.eye(4,4))
        assert_equal(xform_codes.niistring[img.darrays[0].coordsys.dataspace],'NIFTI_XFORM_TALAIRACH')
        assert_equal(xform_codes.niistring[img.darrays[0].coordsys.xformspace],'NIFTI_XFORM_TALAIRACH')
'''


def test_readwritedata():
    with InTemporaryDirectory():
        for name in datafiles:
            img = nib.load(name)
            nib.save(img, 'test.nii')
            img2 = nib.load('test.nii')
            assert_equal(len(img.header.matrix.mims), len(img2.header.matrix.mims))
            assert_array_almost_equal(img.data,
                                      img2.data)
