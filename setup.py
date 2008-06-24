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

from distutils.core import setup, Extension
import os
import numpy as N
from glob import glob

nifti_wrapper_file = os.path.join('nifti', 'nifticlib.py')

# create an empty file to workaround crappy swig wrapper installation
if not os.path.isfile(nifti_wrapper_file):
    open(nifti_wrapper_file, 'w')

# find numpy headers
numpy_headers = os.path.join(os.path.dirname(N.__file__),'core','include')


# Notes on the setup
# Version scheme is:
# 0.<4-digit-year><2-digit-month><2-digit-day>.<ever-increasing-integer>

setup(name       = 'pynifti',
    version      = '0.20080624.1',
    author       = 'Michael Hanke',
    author_email = 'michael.hanke@gmail.com',
    license      = 'MIT License',
    url          = 'http://niftilib.sf.net/pynifti',
    description  = 'Python interface for the NIfTI IO libraries',
    long_description = """ """,
    packages     = [ 'nifti' ],
    scripts      = glob( 'bin/*' ),
    ext_modules  = [ Extension( 'nifti._nifticlib', [ 'nifti/nifticlib.i' ],
            include_dirs = [ '/usr/include/nifti', numpy_headers ],
            libraries    = [ 'niftiio' ],
            swig_opts    = [ '-I/usr/include/nifti',
                             '-I' + numpy_headers ] ) ]
    )

