import os
import unittest
import numpy as np

from nibabel.externals.six import BytesIO

from nibabel.testing import suppress_warnings, catch_warn_reset
from nibabel.testing import assert_arrays_equal, assert_streamlines_equal
from nose.tools import assert_equal, assert_raises, assert_true

from .. import base_format
from ..base_format import Streamlines, LazyStreamlines
from ..base_format import DataError, HeaderError, HeaderWarning, UsageWarning

from .. import trk
from ..trk import TrkFile

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def isiterable(streamlines):
    try:
        for point, scalar, prop in streamlines:
            pass
    except:
        return False

    return True


def check_streamlines(streamlines, nb_streamlines, points, scalars, properties):
    # Check data
    assert_equal(len(streamlines), nb_streamlines)
    assert_arrays_equal(streamlines.points, points)
    assert_arrays_equal(streamlines.scalars, scalars)
    assert_arrays_equal(streamlines.properties, properties)
    assert_true(isiterable(streamlines))

    assert_equal(streamlines.header.nb_streamlines, nb_streamlines)
    nb_scalars_per_point = 0 if len(scalars) == 0 else len(scalars[0][0])
    nb_properties_per_streamline = 0 if len(properties) == 0 else len(properties[0])
    assert_equal(streamlines.header.nb_scalars_per_point, nb_scalars_per_point)
    assert_equal(streamlines.header.nb_properties_per_streamline, nb_properties_per_streamline)


class TestTRK(unittest.TestCase):

    def setUp(self):
        self.empty_trk_filename = os.path.join(DATA_PATH, "empty.trk")
        # simple.trk contains only points
        self.simple_trk_filename = os.path.join(DATA_PATH, "simple.trk")
        # complex.trk contains points, scalars and properties
        self.complex_trk_filename = os.path.join(DATA_PATH, "complex.trk")

        self.points = [np.arange(1*3, dtype="f4").reshape((1, 3)),
                       np.arange(2*3, dtype="f4").reshape((2, 3)),
                       np.arange(5*3, dtype="f4").reshape((5, 3))]

        self.colors = [np.array([(1, 0, 0)]*1, dtype="f4"),
                       np.array([(0, 1, 0)]*2, dtype="f4"),
                       np.array([(0, 0, 1)]*5, dtype="f4")]

        self.mean_curvature_torsion = [np.array([1.11, 1.22], dtype="f4"),
                                       np.array([2.11, 2.22], dtype="f4"),
                                       np.array([3.11, 3.22], dtype="f4")]

        self.nb_streamlines = len(self.points)
        self.nb_scalars_per_point = self.colors[0].shape[1]
        self.nb_properties_per_streamline = len(self.mean_curvature_torsion[0])

    def test_load_empty_file(self):
        trk = TrkFile.load(self.empty_trk_filename, ref=None, lazy_load=False)
        check_streamlines(trk, 0, [], [], [])

        trk = TrkFile.load(self.empty_trk_filename, ref=None, lazy_load=True)
        # Suppress warning about loading a TRK file in lazy mode with count=0.
        with suppress_warnings():
            check_streamlines(trk, 0, [], [], [])

    def test_load_simple_file(self):
        trk = TrkFile.load(self.simple_trk_filename, ref=None, lazy_load=False)
        check_streamlines(trk, self.nb_streamlines, self.points, [], [])

        trk = TrkFile.load(self.simple_trk_filename, ref=None, lazy_load=True)
        check_streamlines(trk, self.nb_streamlines, self.points, [], [])

    def test_load_complex_file(self):
        trk = TrkFile.load(self.complex_trk_filename, ref=None, lazy_load=False)
        check_streamlines(trk, self.nb_streamlines,
                          self.points, self.colors, self.mean_curvature_torsion)

        trk = TrkFile.load(self.complex_trk_filename, ref=None, lazy_load=True)
        check_streamlines(trk, self.nb_streamlines,
                          self.points, self.colors, self.mean_curvature_torsion)

    def test_load_file_with_wrong_information(self):
        trk_file = open(self.simple_trk_filename, 'rb').read()

        # Simulate a TRK file where `count` was not provided.
        count = np.array(0, dtype="int32").tostring()
        new_trk_file = trk_file[:1000-12] + count + trk_file[1000-8:]
        streamlines = TrkFile.load(BytesIO(new_trk_file), lazy_load=False)
        check_streamlines(streamlines, self.nb_streamlines, self.points, [], [])

        streamlines = TrkFile.load(BytesIO(new_trk_file), lazy_load=True)
        with catch_warn_reset(record=True, modules=[base_format]) as w:
            check_streamlines(streamlines, self.nb_streamlines, self.points, [], [])
            assert_equal(len(w), 1)
            assert_true(issubclass(w[0].category, UsageWarning))

        # Simulate a TRK file where `voxel_order` was not provided.
        voxel_order = np.zeros(1, dtype="|S3").tostring()
        new_trk_file = trk_file[:948] + voxel_order + trk_file[948+3:]
        with catch_warn_reset(record=True, modules=[trk]) as w:
            TrkFile.load(BytesIO(new_trk_file), ref=None)
            assert_equal(len(w), 1)
            assert_true(issubclass(w[0].category, HeaderWarning))
            assert_true("LPS" in str(w[0].message))

        # Simulate a TRK file with an unsupported version.
        version = np.int32(123).tostring()
        new_trk_file = trk_file[:992] + version + trk_file[992+4:]
        assert_raises(HeaderError, TrkFile.load, BytesIO(new_trk_file))

        # Simulate a TRK file with a wrong hdr_size.
        hdr_size = np.int32(1234).tostring()
        new_trk_file = trk_file[:996] + hdr_size + trk_file[996+4:]
        assert_raises(HeaderError, TrkFile.load, BytesIO(new_trk_file))

    def test_write_simple_file(self):
        streamlines = Streamlines(self.points)

        trk_file = BytesIO()
        TrkFile.save(streamlines, trk_file)
        trk_file.seek(0, os.SEEK_SET)

        loaded_streamlines = TrkFile.load(trk_file)
        check_streamlines(loaded_streamlines, self.nb_streamlines,
                          self.points, [], [])

        loaded_streamlines_orig = TrkFile.load(self.simple_trk_filename)
        assert_streamlines_equal(loaded_streamlines, loaded_streamlines_orig)

        trk_file.seek(0, os.SEEK_SET)
        assert_equal(open(self.simple_trk_filename, 'rb').read(), trk_file.read())

    def test_write_complex_file(self):
        # With scalars
        streamlines = Streamlines(self.points, scalars=self.colors)

        trk_file = BytesIO()
        TrkFile.save(streamlines, trk_file)
        trk_file.seek(0, os.SEEK_SET)

        loaded_streamlines = TrkFile.load(trk_file, ref=None, lazy_load=False)

        check_streamlines(loaded_streamlines, self.nb_streamlines,
                          self.points, self.colors, [])

        # With properties
        streamlines = Streamlines(self.points, properties=self.mean_curvature_torsion)

        trk_file = BytesIO()
        TrkFile.save(streamlines, trk_file)
        trk_file.seek(0, os.SEEK_SET)

        loaded_streamlines = TrkFile.load(trk_file, ref=None, lazy_load=False)
        check_streamlines(loaded_streamlines, self.nb_streamlines,
                          self.points, [], self.mean_curvature_torsion)

        # With scalars and properties
        streamlines = Streamlines(self.points, scalars=self.colors, properties=self.mean_curvature_torsion)

        trk_file = BytesIO()
        TrkFile.save(streamlines, trk_file)
        trk_file.seek(0, os.SEEK_SET)

        loaded_streamlines = TrkFile.load(trk_file, ref=None, lazy_load=False)
        check_streamlines(loaded_streamlines, self.nb_streamlines,
                          self.points, self.colors, self.mean_curvature_torsion)

        loaded_streamlines_orig = TrkFile.load(self.complex_trk_filename)
        assert_streamlines_equal(loaded_streamlines, loaded_streamlines_orig)

        trk_file.seek(0, os.SEEK_SET)
        assert_equal(open(self.complex_trk_filename, 'rb').read(), trk_file.read())

    def test_write_erroneous_file(self):
        # No scalars for every points
        scalars = [[(1, 0, 0)],
                   [(0, 1, 0)],
                   [(0, 0, 1)]]

        streamlines = Streamlines(self.points, scalars)
        assert_raises(DataError, TrkFile.save, streamlines, BytesIO())

        # No scalars for every streamlines
        scalars = [[(1, 0, 0)]*1,
                   [(0, 1, 0)]*2]

        streamlines = Streamlines(self.points, scalars)
        assert_raises(DataError, TrkFile.save, streamlines, BytesIO())

        # Inconsistent number of scalars between points
        scalars = [[(1, 0, 0)]*1,
                   [(0, 1, 0), (0, 1)],
                   [(0, 0, 1)]*5]

        streamlines = Streamlines(self.points, scalars)
        assert_raises(ValueError, TrkFile.save, streamlines, BytesIO())

        # Inconsistent number of scalars between streamlines
        scalars = [[(1, 0, 0)]*1,
                   [(0, 1)]*2,
                   [(0, 0, 1)]*5]

        streamlines = Streamlines(self.points, scalars)
        assert_raises(DataError, TrkFile.save, streamlines, BytesIO())

        # Inconsistent number of properties
        properties = [np.array([1.11, 1.22], dtype="f4"),
                      np.array([2.11], dtype="f4"),
                      np.array([3.11, 3.22], dtype="f4")]
        streamlines = Streamlines(self.points, properties=properties)
        assert_raises(DataError, TrkFile.save, streamlines, BytesIO())

        # No properties for every streamlines
        properties = [np.array([1.11, 1.22], dtype="f4"),
                      np.array([2.11, 2.22], dtype="f4")]
        streamlines = Streamlines(self.points, properties=properties)
        assert_raises(DataError, TrkFile.save, streamlines, BytesIO())

    def test_write_file_lazy_streamlines(self):
        points = lambda: (point for point in self.points)
        scalars = lambda: (scalar for scalar in self.colors)
        properties = lambda: (prop for prop in self.mean_curvature_torsion)

        streamlines = LazyStreamlines(points, scalars, properties)
        # No need to manually set `nb_streamlines` in the header since we count
        # them as writing.
        #streamlines.header.nb_streamlines = self.nb_streamlines

        trk_file = BytesIO()
        TrkFile.save(streamlines, trk_file)
        trk_file.seek(0, os.SEEK_SET)

        trk = TrkFile.load(trk_file, ref=None, lazy_load=False)
        check_streamlines(trk, self.nb_streamlines,
                          self.points, self.colors, self.mean_curvature_torsion)
