#!/usr/bin/env python
# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Print nifti diagnostics '''

import sys

import nibabel

for fname in sys.argv[1:]:
    fobj = nibabel.volumeutils.allopen(fname)
    hdr =  fobj.read(nibabel.nifti1.header_dtype.itemsize)
    print 'Picky header check output for "%s"\n' % fname
    print nibabel.Nifti1Header.diagnose_binaryblock(hdr)
    print
