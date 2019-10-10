import sys
import os
from .optpkg import optional_package

# PY35: A bug affected Windows installations of h5py in Python3 versions <3.6
# due to random dictionary ordering, causing float64 data arrays to sometimes be
# loaded as longdouble (also 64 bit on Windows). This caused stochastic failures
# to correctly handle data caches, and possibly other subtle bugs we never
# caught. This was fixed in h5py 2.10.
# Please see https://github.com/nipy/nibabel/issues/665 for details.
min_h5py = '2.10' if os.name == 'nt' and (3,) <= sys.version_info < (3, 6) else None
h5py, have_h5py, setup_module = optional_package('h5py', min_version=min_h5py)
