""" Benchmarks for array_to_file routine

Run benchmarks with::

    import nibabel as nib
    nib.bench()

If you have doctests enabled by default in nose (with a noserc file or
environment variable), and you have a numpy version <= 1.6.1, this will also
run the doctests, let's hope they pass.

Run this benchmark with:

    nosetests -s --match '(?:^|[\\b_\\.//-])[Bb]ench' /path/to/bench_load_save.py
"""
from __future__ import division, print_function

import sys

import numpy as np


from .butils import print_git_title

from numpy.testing import measure


def bench_array_to_file():
    rng = np.random.RandomState(20111001)
    repeat = 10
    img_shape = (128, 128, 64, 10)
    arr = rng.normal(size=img_shape)
    sys.stdout.flush()
    print_git_title("\nArray to file")
    mtime = measure('array_to_file(arr, BytesIO(), np.float32)', repeat)
    print('%30s %6.2f' % ('Save float64 to float32', mtime))
    mtime = measure('array_to_file(arr, BytesIO(), np.int16)', repeat)
    print('%30s %6.2f' % ('Save float64 to int16', mtime))
    # Set a lot of NaNs to check timing
    arr[:, :, :, 1] = np.nan
    mtime = measure('array_to_file(arr, BytesIO(), np.float32)', repeat)
    print('%30s %6.2f' % ('Save float64 to float32, NaNs', mtime))
    mtime = measure('array_to_file(arr, BytesIO(), np.int16)', repeat)
    print('%30s %6.2f' % ('Save float64 to int16, NaNs', mtime))
    # Set a lot of infs to check timing
    arr[:, :, :, 1] = np.inf
    mtime = measure('array_to_file(arr, BytesIO(), np.float32)', repeat)
    print('%30s %6.2f' % ('Save float64 to float32, infs', mtime))
    mtime = measure('array_to_file(arr, BytesIO(), np.int16)', repeat)
    print('%30s %6.2f' % ('Save float64 to int16, infs', mtime))
    # Int16 input, float output
    arr = np.random.random_integers(low=-1000, high=1000, size=img_shape)
    arr = arr.astype(np.int16)
    mtime = measure('array_to_file(arr, BytesIO(), np.float32)', repeat)
    print('%30s %6.2f' % ('Save Int16 to float32', mtime))
    sys.stdout.flush()
