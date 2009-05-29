#!/usr/bin/env python
''' Print nifti diagnostics '''

import sys

import nifti

for fname in sys.argv[1:]:
    fobj = nifti.volumeutils.allopen(fname)
    hdr =  fobj.read(nifti.nifti1.header_dtype.itemsize)
    print 'Picky header check output for "%s"\n' % fname
    print nifti.Nifti1Header.diagnose_binaryblock(hdr)
    print
