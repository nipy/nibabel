#!/usr/bin/env python
''' Print nifti diagnostics '''

import sys

import nibabel

for fname in sys.argv[1:]:
    fobj = nibabel.volumeutils.allopen(fname)
    hdr =  fobj.read(nibabel.nifti1.header_dtype.itemsize)
    print 'Picky header check output for "%s"\n' % fname
    print nibabel.Nifti1Header.diagnose_binaryblock(hdr)
    print
