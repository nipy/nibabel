#!/usr/bin/env python

from distutils.core import setup, Extension
import os
import numpy

nifti_wrapper_file = os.path.join('nifti', 'clibs.py')

# create an empty file to workaround crappy swig wrapper installation
if not os.path.isfile(nifti_wrapper_file):
	open(nifti_wrapper_file, 'w')

# find numpy headers
numpy_headers = os.path.join(os.path.dirname(numpy.__file__),'core','include')

setup(name = 'pynifti',
	version = '0.20061116',
	author = 'Michael Hanke',
	author_email = 'michael.hanke@gmail.com',
	license = 'LGPL',
	url = 'http://apsy.gse.uni-magdeburg.de/hanke',
	description = 'Python interface for the NIfTI IO libraries',
	long_description = """ """,
	packages = [ 'nifti' ],
	ext_modules = [ Extension( 'nifti._clibs', 
			['nifti/clibs.i' ], 
			include_dirs = ['/usr/include/nifti', numpy_headers ],
			libraries = [ 'niftiio', 'fslio' ],
			swig_opts = [ '-I/usr/include/nifti',  '-I' + numpy_headers ],
			) ]
      )

