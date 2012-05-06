""" Benchmarks for load and save of image arrays

Run benchmarks with::

    import nibabel as nib
    nib.bench()

If you have doctests enabled by default in nose (with a noserc file or
environment variable), and you have a numpy version <= 1.6.1, this will also run
the doctests, let's hope they pass.
"""
import sys

import numpy as np

from ..py3k import BytesIO
from .. import Nifti1Image

from numpy.testing import measure

def bench_load_save():
    rng = np.random.RandomState(20111001)
    repeat = 4
    img_shape = (128, 128, 64)
    arr = rng.normal(size=img_shape)
    img = Nifti1Image(arr, np.eye(4))
    sio = BytesIO()
    img.file_map['image'].fileobj = sio
    hdr = img.get_header()
    sys.stdout.flush()
    print "\nImage load save"
    print "----------------"
    hdr.set_data_dtype(np.float32)
    mtime = measure('img.to_file_map()', repeat)
    print '%30s %6.2f' % ('Save float64 to float32', mtime)
    mtime = measure('img.from_file_map(img.file_map)', repeat)
    print '%30s %6.2f' % ('Load from float32', mtime)
    hdr.set_data_dtype(np.int16)
    mtime = measure('img.to_file_map()', repeat)
    print '%30s %6.2f' % ('Save float64 to int16', mtime)
    mtime = measure('img.from_file_map(img.file_map)', repeat)
    print '%30s %6.2f' % ('Load from int16', mtime)
    arr = np.random.random_integers(low=-1000,high=-1000, size=img_shape)
    arr = arr.astype(np.int16)
    img = Nifti1Image(arr, np.eye(4))
    sio = BytesIO()
    img.file_map['image'].fileobj = sio
    hdr = img.get_header()
    hdr.set_data_dtype(np.float32)
    mtime = measure('img.to_file_map()', repeat)
    print '%30s %6.2f' % ('Save Int16 to float32', mtime)
    sys.stdout.flush()
