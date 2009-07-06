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
import tempfile
import shutil

from nose.tools import ok_
from nifti.testing import example_data_path
from nifti.nifti1 import save, load, Nifti1Extension, extension_codes

data_path, _ = os.path.split(__file__)
data_path = os.path.join(data_path, 'data')
image_file = os.path.join(data_path, 'example4d.nii.gz')


def test_extension_codes():
    for k in extension_codes.keys():
        ext = Nifti1Extension(k, 'somevalue')


def test_nifti_extensions():
    nim = load(image_file)
    # basic checks of the available extensions
    ext = nim.extra['extensions']
    ok_(len(ext) == 2)
    ok_(ext.count('comment') == 2)
    ok_(ext.count('afni') == 0)
    ok_(ext.get_codes() == [6, 6])

    # first extension should be short one
    ok_(ext[0].get_content() == 'extcomment1')

    # add one
    afniext = Nifti1Extension('afni', '<xml></xml>')
    ext.append(afniext)
    ok_(ext.get_codes() == [6, 6, 4])
    ok_(ext.count('comment') == 2)
    ok_(ext.count('afni') == 1)

    # delete one
    del ext[1]
    ok_(ext.get_codes() == [6, 4])
    ok_(ext.count('comment') == 1)
    ok_(ext.count('afni') == 1)

class ExtensionIOTests(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp()
        self.nimg = load(image_file)
        self.fp = tempfile.NamedTemporaryFile(suffix='.nii.gz')

    def tearDown(self):
        shutil.rmtree(self.workdir)
        del self.nimg
        self.fp.close()

    def testExtensionLoadSaveCycle(self):
        self.nimg.to_files(self.nimg.filespec_to_files(self.fp.name))
        ok_(self.nimg.extra.has_key('extensions'))
        lnim = load(self.fp.name)
        ok_(lnim.extra.has_key('extensions'))
        print self.nimg.extra['extensions']
        print lnim.extra['extensions']
        ok_(self.nimg.extra['extensions'] == lnim.extra['extensions'])
