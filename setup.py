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

import os

import setuptools

# Commit hash writing
from nisext.sexts import get_comrec_build, read_vars_from

INFO = read_vars_from(os.path.join('nibabel', 'info.py'))

if __name__ == "__main__":
    setuptools.setup(name='nibabel',
                     version=INFO.VERSION,
                     setup_requires=['setuptools>=30.3.0'],
                     cmdclass={'build_py': get_comrec_build('nibabel')})
