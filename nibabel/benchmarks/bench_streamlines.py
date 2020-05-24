""" Benchmarks for load and save of streamlines

Run benchmarks with::

    import nibabel as nib
    nib.bench()

Run this benchmark with::

    pytest -c <path>/benchmarks/pytest.benchmark.ini <path>/benchmarks/bench_streamlines.py
"""

import numpy as np

from nibabel.tmpdirs import InTemporaryDirectory

from numpy.testing import assert_array_equal
from nibabel.streamlines import Tractogram
from nibabel.streamlines import TrkFile

import nibabel as nib
import nibabel.trackvis as tv

from numpy.testing import measure


def bench_load_trk():
    rng = np.random.RandomState(42)
    dtype = 'float32'
    NB_STREAMLINES = 5000
    NB_POINTS = 1000
    points = [rng.rand(NB_POINTS, 3).astype(dtype)
              for i in range(NB_STREAMLINES)]
    scalars = [rng.rand(NB_POINTS, 10).astype(dtype)
               for i in range(NB_STREAMLINES)]

    repeat = 10

    with InTemporaryDirectory():
        trk_file = "tmp.trk"
        tractogram = Tractogram(points, affine_to_rasmm=np.eye(4))
        TrkFile(tractogram).save(trk_file)

        streamlines_old = [d[0] - 0.5
                           for d in tv.read(trk_file, points_space="rasmm")[0]]
        mtime_old = measure('tv.read(trk_file, points_space="rasmm")', repeat)
        print(f"Old: Loaded {NB_STREAMLINES:,} streamlines in {mtime_old:6.2f}")

        trk = nib.streamlines.load(trk_file, lazy_load=False)
        streamlines_new = trk.streamlines
        mtime_new = measure('nib.streamlines.load(trk_file, lazy_load=False)',
                            repeat)
        print(f"\nNew: Loaded {NB_STREAMLINES:,} streamlines in {mtime_new:6.2}")
        print(f"Speedup of {mtime_old / mtime_new:.2f}")
        for s1, s2 in zip(streamlines_new, streamlines_old):
            assert_array_equal(s1, s2)

    # Points and scalars
    with InTemporaryDirectory():

        trk_file = "tmp.trk"
        tractogram = Tractogram(points,
                                data_per_point={'scalars': scalars},
                                affine_to_rasmm=np.eye(4))
        TrkFile(tractogram).save(trk_file)

        streamlines_old = [d[0] - 0.5
                           for d in tv.read(trk_file, points_space="rasmm")[0]]

        scalars_old = [d[1]
                       for d in tv.read(trk_file, points_space="rasmm")[0]]
        mtime_old = measure('tv.read(trk_file, points_space="rasmm")', repeat)
        msg = "Old: Loaded {:,} streamlines with scalars in {:6.2f}"
        print(msg.format(NB_STREAMLINES, mtime_old))

        trk = nib.streamlines.load(trk_file, lazy_load=False)
        scalars_new = trk.tractogram.data_per_point['scalars']
        mtime_new = measure('nib.streamlines.load(trk_file, lazy_load=False)',
                            repeat)
        msg = "New: Loaded {:,} streamlines with scalars in {:6.2f}"
        print(msg.format(NB_STREAMLINES, mtime_new))
        print(f"Speedup of {mtime_old / mtime_new:2f}")
        for s1, s2 in zip(scalars_new, scalars_old):
            assert_array_equal(s1, s2)


if __name__ == '__main__':
    bench_load_trk()
