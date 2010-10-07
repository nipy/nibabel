# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Simple wrapper to use setuptools extension bdist_mpkg with NiBabel
distutils setup.py.

This script is a minimal version of a wrapper script shipped with the
bdist_mpkg packge.
"""

__docformat__ = 'restructuredtext'

import sys
import setuptools
import bdist_mpkg

def main():
    del sys.argv[0]
    sys.argv.insert(1, 'bdist_mpkg')
    g = dict(globals())
    g['__file__'] = sys.argv[0]
    g['__name__'] = '__main__'
    execfile(sys.argv[0], g, g)

if __name__ == '__main__':
    main()
