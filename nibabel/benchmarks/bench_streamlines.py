""" Benchmarks for load and save of streamlines

Run benchmarks with::

    import nibabel as nib
    nib.bench()

If you have doctests enabled by default in nose (with a noserc file or
environment variable), and you have a numpy version <= 1.6.1, this will also run
the doctests, let's hope they pass.

Run this benchmark with:

    nosetests -s --match '(?:^|[\\b_\\.//-])[Bb]ench' /path/to/bench_streamlines.py
"""
from __future__ import division, print_function

import os
import numpy as np

from nibabel.externals.six import BytesIO
from nibabel.externals.six.moves import zip

from nibabel.testing import assert_arrays_equal

from numpy.testing import assert_array_equal
from nibabel.streamlines.base_format import Streamlines
from nibabel.streamlines import TrkFile

import nibabel as nib
import nibabel.trackvis as tv

from numpy.testing import measure


def bench_load_trk():
    NB_STREAMLINES = 1000
    NB_POINTS = 1000
    points = [np.random.rand(NB_POINTS, 3).astype('float32') for i in range(NB_STREAMLINES)]
    repeat = 20

    trk_file = BytesIO()
    #trk = list(zip(points, [None]*NB_STREAMLINES, [None]*NB_STREAMLINES))
    #tv.write(trk_file, trk)
    streamlines = Streamlines(points)
    TrkFile.save(streamlines, trk_file)

    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput

    # with PyCallGraph(output=GraphvizOutput()):
    #     #nib.streamlines.load(trk_file, ref=None, lazy_load=False)

    mtime_new = measure('trk_file.seek(0, os.SEEK_SET); nib.streamlines.load(trk_file, ref=None, lazy_load=False)', repeat)
    print("\nNew: Loaded %d streamlines in %6.2f" % (NB_STREAMLINES, mtime_new))

    mtime_old = measure('trk_file.seek(0, os.SEEK_SET); tv.read(trk_file, points_space="voxel")', repeat)
    print("Old: Loaded %d streamlines in %6.2f" % (NB_STREAMLINES, mtime_old))
    print("Speedup of %2f" % (mtime_old/mtime_new))

    # Points and scalars
    scalars = [np.random.rand(NB_POINTS, 10).astype('float32') for i in range(NB_STREAMLINES)]

    trk_file = BytesIO()
    #trk = list(zip(points, scalars, [None]*NB_STREAMLINES))
    #tv.write(trk_file, trk)
    streamlines = Streamlines(points, scalars)
    TrkFile.save(streamlines, trk_file)

    mtime_new = measure('trk_file.seek(0, os.SEEK_SET); nib.streamlines.load(trk_file, ref=None, lazy_load=False)', repeat)
    print("New: Loaded %d streamlines with scalars in %6.2f" % (NB_STREAMLINES, mtime_new))

    mtime_old = measure('trk_file.seek(0, os.SEEK_SET); tv.read(trk_file, points_space="voxel")', repeat)
    print("Old: Loaded %d streamlines with scalars in %6.2f" % (NB_STREAMLINES, mtime_old))
    print("Speedup of %2f" % (mtime_old/mtime_new))


def bench_save_trk():
    NB_STREAMLINES = 100
    NB_POINTS = 1000
    points = [np.random.rand(NB_POINTS, 3).astype('float32') for i in range(NB_STREAMLINES)]
    repeat = 10

    # Only points
    streamlines = Streamlines(points)
    trk_file_new = BytesIO()

    mtime_new = measure('trk_file_new.seek(0, os.SEEK_SET); TrkFile.save(streamlines, trk_file_new)', repeat)
    print("\nNew: Saved %d streamlines in %6.2f" % (NB_STREAMLINES, mtime_new))

    trk_file_old = BytesIO()
    trk = list(zip(points, [None]*NB_STREAMLINES, [None]*NB_STREAMLINES))
    mtime_old = measure('trk_file_old.seek(0, os.SEEK_SET); tv.write(trk_file_old, trk)', repeat)
    print("Old: Saved %d streamlines in %6.2f" % (NB_STREAMLINES, mtime_old))
    print("Speedup of %2f" % (mtime_old/mtime_new))

    trk_file_new.seek(0, os.SEEK_SET)
    trk_file_old.seek(0, os.SEEK_SET)
    streams, hdr = tv.read(trk_file_old)

    for pts, A in zip(points, streams):
        assert_array_equal(pts, A[0])

    trk = nib.streamlines.load(trk_file_new, ref=None, lazy_load=False)
    assert_arrays_equal(points, trk.points)

    # Points and scalars
    scalars = [np.random.rand(NB_POINTS, 3).astype('float32') for i in range(NB_STREAMLINES)]
    streamlines = Streamlines(points, scalars=scalars)
    trk_file_new = BytesIO()

    mtime_new = measure('trk_file_new.seek(0, os.SEEK_SET); TrkFile.save(streamlines, trk_file_new)', repeat)
    print("New: Saved %d streamlines with scalars in %6.2f" % (NB_STREAMLINES, mtime_new))

    trk_file_old = BytesIO()
    trk = list(zip(points, scalars, [None]*NB_STREAMLINES))
    mtime_old = measure('trk_file_old.seek(0, os.SEEK_SET); tv.write(trk_file_old, trk)', repeat)
    print("Old: Saved %d streamlines with scalars in %6.2f" % (NB_STREAMLINES, mtime_old))
    print("Speedup of %2f" % (mtime_old/mtime_new))

    trk_file_new.seek(0, os.SEEK_SET)
    trk_file_old.seek(0, os.SEEK_SET)
    streams, hdr = tv.read(trk_file_old)

    for pts, scal, A in zip(points, scalars, streams):
        assert_array_equal(pts, A[0])
        assert_array_equal(scal, A[1])

    trk = nib.streamlines.load(trk_file_new, ref=None, lazy_load=False)

    assert_arrays_equal(points, trk.points)
    assert_arrays_equal(scalars, trk.scalars)


if __name__ == '__main__':
    bench_save_trk()
