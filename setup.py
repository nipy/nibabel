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

import sys
import os

from setuptools import setup

# nisext is nipy setup extensions, which we're mostly moving away from
# get_comrec_build stores the current commit in COMMIT_HASH.txt at build time
# read_vars_from evaluates a python file and makes variables available
from nisext.sexts import get_comrec_build, read_vars_from

INFO = read_vars_from(os.path.join('nibabel', 'info.py'))

# Give setuptools a hint to complain if it's too old a version
# 30.3.0 allows us to put most metadata in setup.cfg
# Should match pyproject.toml
SETUP_REQUIRES = ['setuptools >= 30.3.0']
# This enables setuptools to install wheel on-the-fly
SETUP_REQUIRES += ['wheel'] if 'bdist_wheel' in sys.argv else []

if __name__ == "__main__":
    setup(name='nibabel',
          version=INFO.VERSION,
          setup_requires=SETUP_REQUIRES,
          cmdclass={'build_py': get_comrec_build('nibabel')})
