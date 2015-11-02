import os
import unittest
import numpy as np

from nibabel.externals.six import BytesIO

from nibabel.testing import suppress_warnings, clear_and_catch_warnings
from nibabel.testing import assert_arrays_equal, assert_tractogram_equal, isiterable
from nose.tools import assert_equal, assert_raises, assert_true

from .. import base_format
from ..base_format import Tractogram, LazyTractogram
from ..base_format import DataError, HeaderError, HeaderWarning, UsageWarning

#from .. import trk
from ..trk import TrkFile

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def check_tractogram(tractogram, nb_streamlines, streamlines, scalars, properties):
    # Check data
    assert_equal(len(tractogram), nb_streamlines)
    assert_arrays_equal(tractogram.streamlines, streamlines)
    assert_arrays_equal(tractogram.scalars, scalars)
    assert_arrays_equal(tractogram.properties, properties)
    assert_true(isiterable(tractogram))

    assert_equal(tractogram.header.nb_streamlines, nb_streamlines)
    nb_scalars_per_point = 0 if len(scalars) == 0 else len(scalars[0][0])
    nb_properties_per_streamline = 0 if len(properties) == 0 else len(properties[0])
    assert_equal(tractogram.header.nb_scalars_per_point, nb_scalars_per_point)
    assert_equal(tractogram.header.nb_properties_per_streamline, nb_properties_per_streamline)


class TestTRK(unittest.TestCase):

    def setUp(self):
        self.empty_trk_filename = os.path.join(DATA_PATH, "empty.trk")
        # simple.trk contains only streamlines
        self.simple_trk_filename = os.path.join(DATA_PATH, "simple.trk")
        # complex.trk contains streamlines, scalars and properties
        self.complex_trk_filename = os.path.join(DATA_PATH, "complex.trk")

        self.streamlines = [np.arange(1*3, dtype="f4").reshape((1, 3)),
                            np.arange(2*3, dtype="f4").reshape((2, 3)),
                            np.arange(5*3, dtype="f4").reshape((5, 3))]

        self.colors = [np.array([(1, 0, 0)]*1, dtype="f4"),
                       np.array([(0, 1, 0)]*2, dtype="f4"),
                       np.array([(0, 0, 1)]*5, dtype="f4")]

        self.mean_curvature_torsion = [np.array([1.11, 1.22], dtype="f4"),
                                       np.array([2.11, 2.22], dtype="f4"),
                                       np.array([3.11, 3.22], dtype="f4")]

        self.nb_streamlines = len(self.streamlines)
        self.nb_scalars_per_point = self.colors[0].shape[1]
        self.nb_properties_per_streamline = len(self.mean_curvature_torsion[0])
        self.affine = np.eye(4)

    def test_load_empty_file(self):
        trk = TrkFile.load(self.empty_trk_filename, ref=None, lazy_load=False)
        check_tractogram(trk.tractogram, 0, [], [], [])

        trk = TrkFile.load(self.empty_trk_filename, ref=None, lazy_load=True)
        # Suppress warning about loading a TRK file in lazy mode with count=0.
        with suppress_warnings():
            check_tractogram(trk.tractogram, 0, [], [], [])

    def test_load_simple_file(self):
        trk = TrkFile.load(self.simple_trk_filename, ref=None, lazy_load=False)
        check_tractogram(trk.tractogram, self.nb_streamlines, self.streamlines, [], [])

        trk = TrkFile.load(self.simple_trk_filename, ref=None, lazy_load=True)
        check_tractogram(trk.tractogram, self.nb_streamlines, self.streamlines, [], [])

    def test_load_complex_file(self):
        trk = TrkFile.load(self.complex_trk_filename, ref=None, lazy_load=False)
        check_tractogram(trk.tractogram, self.nb_streamlines,
                          self.streamlines, self.colors, self.mean_curvature_torsion)

        trk = TrkFile.load(self.complex_trk_filename, ref=None, lazy_load=True)
        check_tractogram(trk.tractogram, self.nb_streamlines,
                          self.streamlines, self.colors, self.mean_curvature_torsion)

    def test_load_file_with_wrong_information(self):
        trk_file = open(self.simple_trk_filename, 'rb').read()

        # Simulate a TRK file where `count` was not provided.
        count = np.array(0, dtype="int32").tostring()
        new_trk_file = trk_file[:1000-12] + count + trk_file[1000-8:]
        trk = TrkFile.load(BytesIO(new_trk_file), lazy_load=False)
        check_tractogram(trk.tractogram, self.nb_streamlines, self.streamlines, [], [])

        trk = TrkFile.load(BytesIO(new_trk_file), lazy_load=True)
        with clear_and_catch_warnings(record=True, modules=[base_format]) as w:
            check_tractogram(trk.tractogram, self.nb_streamlines, self.streamlines, [], [])
            assert_equal(len(w), 1)
            assert_true(issubclass(w[0].category, UsageWarning))

        # Simulate a TRK file where `voxel_order` was not provided.
        voxel_order = np.zeros(1, dtype="|S3").tostring()
        new_trk_file = trk_file[:948] + voxel_order + trk_file[948+3:]
        with clear_and_catch_warnings(record=True, modules=[trk]) as w:
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
        tractogram = Tractogram(self.streamlines)

        trk_file = BytesIO()
        trk = TrkFile(tractogram, ref=self.affine)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)

        loaded_trk = TrkFile.load(trk_file)
        check_tractogram(loaded_trk.tractogram, self.nb_streamlines,
                         self.streamlines, [], [])

        loaded_trk_orig = TrkFile.load(self.simple_trk_filename)
        assert_tractogram_equal(loaded_trk.tractogram, loaded_trk_orig.tractogram)

        trk_file.seek(0, os.SEEK_SET)
        assert_equal(open(self.simple_trk_filename, 'rb').read(), trk_file.read())

    def test_write_complex_file(self):
        # With scalars
        tractogram = Tractogram(self.streamlines, scalars=self.colors)

        trk_file = BytesIO()
        trk = TrkFile(tractogram, ref=self.affine)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)

        loaded_trk = TrkFile.load(trk_file, lazy_load=False)
        check_tractogram(loaded_trk.tractogram, self.nb_streamlines,
                         self.streamlines, self.colors, [])

        # With properties
        tractogram = Tractogram(self.streamlines, properties=self.mean_curvature_torsion)

        trk_file = BytesIO()
        trk = TrkFile(tractogram, ref=self.affine)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)

        loaded_trk = TrkFile.load(trk_file, lazy_load=False)
        check_tractogram(loaded_trk.tractogram, self.nb_streamlines,
                         self.streamlines, [], self.mean_curvature_torsion)

        # With scalars and properties
        tractogram = Tractogram(self.streamlines, scalars=self.colors, properties=self.mean_curvature_torsion)

        trk_file = BytesIO()
        trk = TrkFile(tractogram, ref=self.affine)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)

        loaded_trk = TrkFile.load(trk_file, lazy_load=False)
        check_tractogram(loaded_trk.tractogram, self.nb_streamlines,
                         self.streamlines, self.colors, self.mean_curvature_torsion)

        loaded_trk_orig = TrkFile.load(self.complex_trk_filename)
        assert_tractogram_equal(loaded_trk.tractogram, loaded_trk_orig.tractogram)

        trk_file.seek(0, os.SEEK_SET)
        assert_equal(open(self.complex_trk_filename, 'rb').read(), trk_file.read())

    def test_write_erroneous_file(self):
        # No scalars for every points
        scalars = [[(1, 0, 0)],
                   [(0, 1, 0)],
                   [(0, 0, 1)]]

        tractogram = Tractogram(self.streamlines, scalars)
        trk = TrkFile(tractogram, ref=self.affine)
        assert_raises(DataError, trk.save, BytesIO())

        # No scalars for every streamlines
        scalars = [[(1, 0, 0)]*1,
                   [(0, 1, 0)]*2]

        tractogram = Tractogram(self.streamlines, scalars)
        trk = TrkFile(tractogram, ref=self.affine)
        assert_raises(DataError, trk.save, BytesIO())

        # # Unit test moved to test_base_format.py
        # # Inconsistent number of scalars between points
        # scalars = [[(1, 0, 0)]*1,
        #            [(0, 1, 0), (0, 1)],
        #            [(0, 0, 1)]*5]

        # tractogram = Tractogram(self.streamlines, scalars)
        # assert_raises(ValueError, TrkFile.save, tractogram, BytesIO())

        # # Unit test moved to test_base_format.py
        # # Inconsistent number of scalars between streamlines
        # scalars = [[(1, 0, 0)]*1,
        #            [(0, 1)]*2,
        #            [(0, 0, 1)]*5]

        # tractogram = Tractogram(self.streamlines, scalars)
        # assert_raises(DataError, TrkFile.save, tractogram, BytesIO())

        # Unit test moved to test_base_format.py
        # Inconsistent number of properties
        properties = [np.array([1.11, 1.22], dtype="f4"),
                      np.array([2.11], dtype="f4"),
                      np.array([3.11, 3.22], dtype="f4")]
        tractogram = Tractogram(self.streamlines, properties=properties)
        trk = TrkFile(tractogram, ref=self.affine)
        assert_raises(DataError, trk.save, BytesIO())

        # Unit test moved to test_base_format.py
        # No properties for every streamlines
        properties = [np.array([1.11, 1.22], dtype="f4"),
                      np.array([2.11, 2.22], dtype="f4")]
        tractogram = Tractogram(self.streamlines, properties=properties)
        trk = TrkFile(tractogram, ref=self.affine)
        assert_raises(DataError, trk.save, BytesIO())

    # def test_write_file_lazy_tractogram(self):
    #     streamlines = lambda: (point for point in self.streamlines)
    #     scalars = lambda: (scalar for scalar in self.colors)
    #     properties = lambda: (prop for prop in self.mean_curvature_torsion)

    #     tractogram = LazyTractogram(streamlines, scalars, properties)
    #     # No need to manually set `nb_streamlines` in the header since we count
    #     # them as writing.
    #     #tractogram.header.nb_streamlines = self.nb_streamlines

    #     trk_file = BytesIO()
    #     trk = TrkFile(tractogram, ref=self.affine)
    #     trk.save(trk_file)
    #     trk_file.seek(0, os.SEEK_SET)

    #     trk = TrkFile.load(trk_file, ref=None, lazy_load=False)
    #     check_tractogram(trk.tractogram, self.nb_streamlines,
    #                       self.streamlines, self.colors, self.mean_curvature_torsion)
