#!/usr/bin/env python
''' Print nifti diagnostics '''

import sys

import volumeimages.volumeutils as vu
import volumeimages.nifti1 as vn

for fname in sys.argv[1:]:
    fobj = vu.allopen(fname)
    hdr =  fobj.read(vn.header_dtype.itemsize)
    print 'Picky header check output for "%s"\n' % fname
    print vn.Nifti1Header.diagnose_binaryblock(hdr)
    print
