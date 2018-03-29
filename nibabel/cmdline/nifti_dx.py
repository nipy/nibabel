#!python
# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Print nifti diagnostics for header files '''

import sys
from optparse import OptionParser

import nibabel as nib

__author__ = 'Matthew Brett, Yaroslav Halchenko and Christopher Cheng'
__copyright__ = 'Copyright (c) 2018 Matthew Brett ' \
                'and NiBabel contributors'
__license__ = 'MIT'


def main(args=None):
    """ Go go team """
    parser = OptionParser(
        usage="%s [FILE ...]\n\n" % sys.argv[0] + __doc__,
        version="%prog " + nib.__version__)
    (opts, files) = parser.parse_args(args=args)

    for fname in files:
        with nib.openers.ImageOpener(fname) as fobj:
            hdr = fobj.read(nib.nifti1.header_dtype.itemsize)
        result = nib.Nifti1Header.diagnose_binaryblock(hdr)
        if len(result):
            print('Picky header check output for "%s"\n' % fname)
            print(result + '\n')
        else:
            print('Header for "%s" is clean' % fname)