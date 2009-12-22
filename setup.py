#!/usr/bin/env python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Build helper."""

__docformat__ = 'restructuredtext'

import os.path
import sys
from glob import glob

from numpy.distutils.core import setup


# Notes on the setup
# Version scheme is:
# 0.<4-digit-year><2-digit-month><2-digit-day>.<ever-increasing-integer>

setup(name       = 'nibabel',
    version      = '0.2009xxxx.1',
    author       = 'Matthew Brett and Michael Hanke',
    author_email = 'PyNIfTI List <pkg-exppsy-pynifti@lists.alioth.debian.org>',
    license      = 'MIT License',
    url          = 'http://niftilib.sf.net/pynifti',
    description  = 'Python interface for the NIfTI IO libraries',
    long_description = \
        "PyNIfTI aims to provide easy access to NIfTI images from within " \
        "Python. It uses SWIG-generated wrappers for the NIfTI reference " \
        "library and provides the NiftiImage class for Python-style " \
        "access to the image data.\n" \
        "While PyNIfTI is not yet complete (i.e. doesn't support " \
        "everything the C library can do), it already provides access to " \
        "the most important features of the NIfTI-1 data format and " \
        "libniftiio capabilities.",
    packages     = ['nifti'],
    scripts      = glob('bin/*'),
    )
