#!/usr/bin/env python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Build helper."""

__docformat__ = 'restructuredtext'

from numpy.distutils.core import setup
from glob import glob
import os

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
    )
