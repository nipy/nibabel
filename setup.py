#!/usr/bin/env python
# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Build helper."""

__docformat__ = 'restructuredtext'

import os
import sys
from glob import glob

from distutils.core import setup

# For some commands, use setuptools.  
if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb',
            'bdist_wininst', 'install_egg_info', 'egg_info', 'easy_install',
            )).intersection(sys.argv)) > 0:
    # setup_egg imports setuptools setup, thus monkeypatching distutils. 
    from setup_egg import extra_setuptools_args

# extra_setuptools_args can be defined from the line above, but it can
# also be defined here because setup.py has been exec'ed from
# setup_egg.py.
if not 'extra_setuptools_args' in globals():
    extra_setuptools_args = dict()


def main(**extra_args):
    setup(name       = 'nibabel',
          version      = '1.0.0',
          author       = 'Matthew Brett and Michael Hanke',
          author_email = 'NiBabel List <pkg-exppsy-pynifti@lists.alioth.debian.org>',
          license      = 'MIT License',
          url          = 'http://niftilib.sf.net/pynifti',
          description  = 'Access a multitude of neuroimaging data formats',
          long_description = "",
          packages     = ['nibabel',
                          'nibabel.externals',
                          'nibabel.testing',
                          'nibabel.tests'],
          data_files   = [('nibabel/tests/data',
                           glob(os.path.join('nibabel', 'tests', 'data', '*')))],
          scripts      = [os.path.join('bin', 'parrec2nii')],
          **extra_args
         )


if __name__ == "__main__":
    main(**extra_setuptools_args)
