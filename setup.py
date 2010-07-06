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


from numpy.distutils.core import setup, Extension
import os.path
import sys
from glob import glob

########################
# Common configuration #
########################

extra_link_args = ['--Wl,--no-undefined']
include_dirs = []
library_dirs = []
defines = []
link_libs = []
# for some additional swig flags, but see below
swig_opts = []
# a more reliable way to pass options to SWIG
os.environ['SWIG_FEATURES'] = '-O -v'


#############################
# Check for 3rd party stuff #
#############################

# make use of the local nifticlibs copy, only if it was compiled before
if os.path.exists(os.path.join('build', 'nifticlibs', 'libniftiio.a')):
    include_dirs += [os.path.join('3rd', 'nifticlibs')]
    library_dirs += [os.path.join('build', 'nifticlibs')]
    # need to link against additional libs in case of the local static lib
    link_libs += ['znz', 'z']
else:
    # try to look for nifticlibs in some place
    if not sys.platform.startswith('win'):
        include_dirs += ['/usr/include/nifti',
                         '/usr/include/nifticlibs',
                         '/usr/local/include/nifti',
                         '/usr/local/include/nifticlibs',
                         '/usr/local/include']
    else:
        # no clue on windows
        pass


###########################
# Platform-specific setup #
###########################

# win32 stuff
if sys.platform.startswith('win'):
    os.environ['SWIG_FEATURES'] = '-DWIN32 ' + os.environ['SWIG_FEATURES']
    defines.append(('WIN32', None))

# apple stuff
if sys.platform == "darwin":
    extra_link_args.append("-bundle")


##############
# Extensions #
##############

nifticlib_ext = Extension(
    'nifti._clib',
    sources = ['nifti/clib.i'],
    define_macros = defines,
    include_dirs = include_dirs,
    library_dirs = library_dirs,
    libraries = ['niftiio'] + link_libs,
    extra_link_args = extra_link_args,
    swig_opts = swig_opts)

# Notes on the setup
# Version scheme is:
# 0.<4-digit-year><2-digit-month><2-digit-day>.<ever-increasing-integer>

setup(name       = 'pynifti',
    version      = '0.20100607.1',
    author       = 'Michael Hanke',
    author_email = 'michael.hanke@gmail.com',
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
    ext_modules  = [nifticlib_ext]
    )
