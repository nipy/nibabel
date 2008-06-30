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
import os.path
import sys
import numpy as N
from glob import glob

nifti_wrapper_file = os.path.join('nifti', 'nifticlib.py')

# create an empty file to workaround crappy swig wrapper installation
if not os.path.isfile(nifti_wrapper_file):
    open(nifti_wrapper_file, 'w')

# find numpy headers
numpy_headers = os.path.join(os.path.dirname(N.__file__),'core','include')

# include directory: numpy for all 
include_dirs = [ numpy_headers ]

#library dirs: nothing by default
library_dirs = []

# determine what libs to link against
link_libs = [ 'niftiio' ]

# we only know that Debian niftiio is properly linked with znzlib and zlib
# use local niftilib copy on all others
if not os.path.exists('/etc/debian_version'):
    link_libs += ['znz', 'z']
    include_dirs.append(os.path.join('3rd', 'nifticlibs'))
    library_dirs.append(os.path.join('3rd', 'nifticlibs'))
else:
    include_dirs.append('/usr/include/nifti')

swig_opts = []
defines = []
# win32 stuff
if sys.platform.startswith('win'):
    swig_opts.append('-DWIN32')
    defines.append(('WIN32', None))

# Notes on the setup
# Version scheme is:
# 0.<4-digit-year><2-digit-month><2-digit-day>.<ever-increasing-integer>

setup(name       = 'pynifti',
    version      = '0.20080630.1',
    author       = 'Michael Hanke',
    author_email = 'michael.hanke@gmail.com',
    license      = 'MIT License',
    url          = 'http://niftilib.sf.net/pynifti',
    description  = 'Python interface for the NIfTI IO libraries',
    long_description = """ """,
    packages     = [ 'nifti' ],
    scripts      = glob( 'bin/*' ),
    ext_modules  = [ Extension( 'nifti._nifticlib', [ 'nifti/nifticlib.i' ],
            define_macros = defines,
            include_dirs = include_dirs,
            library_dirs = library_dirs,
            libraries    = link_libs,
            swig_opts    = swig_opts + ['-I' + d for d in include_dirs ] ) ]
    )

