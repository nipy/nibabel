"""
Check that loading an image does not use up filehandles.
"""
from __future__ import division, print_function, absolute_import

from os.path import join as pjoin
import shutil
from tempfile import mkdtemp
from warnings import warn

import numpy as np

try:
    import resource as res
except ImportError:
    # Not on Unix, guess limit
    SOFT_LIMIT = 512
else:
    SOFT_LIMIT, HARD_LIMIT = res.getrlimit(res.RLIMIT_NOFILE)

from ..loadsave import load, save
from ..nifti1 import Nifti1Image

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)



def test_multiload():
    # Make a tiny image, save, load many times.  If we are leaking filehandles,
    # this will cause us to run out and generate an error
    N = SOFT_LIMIT + 100
    if N > 5000:
        warn('It would take too long to test file handles, aborting')
        return
    arr = np.arange(24).reshape((2, 3, 4))
    img = Nifti1Image(arr, np.eye(4))
    imgs = []
    try:
        tmpdir = mkdtemp()
        fname = pjoin(tmpdir, 'test.img')
        save(img, fname)
        for i in range(N):
            imgs.append(load(fname))
    finally:
        del img, imgs
        shutil.rmtree(tmpdir)
