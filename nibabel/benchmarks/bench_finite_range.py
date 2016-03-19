""" Benchmarks for finite_range routine

Run benchmarks with::

    import nibabel as nib
    nib.bench()

If you have doctests enabled by default in nose (with a noserc file or
environment variable), and you have a numpy version <= 1.6.1, this will also
run the doctests, let's hope they pass.

Run this benchmark with:

    nosetests -s --match '(?:^|[\\b_\\.//-])[Bb]ench' /path/to/bench_finite_range
"""
from __future__ import division, print_function

import sys

import numpy as np


from .butils import print_git_title

from numpy.testing import measure


def bench_finite_range():
    rng = np.random.RandomState(20111001)
    repeat = 10
    img_shape = (128, 128, 64, 10)
    arr = rng.normal(size=img_shape)
    sys.stdout.flush()
    print_git_title("\nFinite range")
    mtime = measure('finite_range(arr)', repeat)
    print('%30s %6.2f' % ('float64 all finite', mtime))
    arr[:, :, :, 1] = np.nan
    mtime = measure('finite_range(arr)', repeat)
    print('%30s %6.2f' % ('float64 many NaNs', mtime))
    arr[:, :, :, 1] = np.inf
    mtime = measure('finite_range(arr)', repeat)
    print('%30s %6.2f' % ('float64 many infs', mtime))
    # Int16 input, float output
    arr = np.random.random_integers(low=-1000, high=1000, size=img_shape)
    arr = arr.astype(np.int16)
    mtime = measure('finite_range(arr)', repeat)
    print('%30s %6.2f' % ('int16', mtime))
    sys.stdout.flush()
