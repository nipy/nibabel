import os
import unittest
import numpy as np

from nibabel.externals.six import BytesIO

from nibabel.testing import suppress_warnings, clear_and_catch_warnings
from nibabel.testing import assert_arrays_equal, check_iteration
from nose.tools import assert_equal, assert_raises, assert_true

from .test_tractogram import assert_tractogram_equal
from ..tractogram import Tractogram, LazyTractogram
from ..tractogram_file import DataError, HeaderError, HeaderWarning

from .. import trk as trk_module
from ..trk import TrkFile, header_2_dtype

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def assert_header_equal(h1, h2):
    header1 = np.zeros(1, dtype=header_2_dtype)
    header2 = np.zeros(1, dtype=header_2_dtype)

    for k, v in h1.items():
        header1[k] = v

    for k, v in h2.items():
        header2[k] = v

    assert_equal(header1, header2)


class TestTRK(unittest.TestCase):

    def setUp(self):
        self.empty_trk_filename = os.path.join(DATA_PATH, "empty.trk")
        # simple.trk contains only streamlines
        self.simple_trk_filename = os.path.join(DATA_PATH, "simple.trk")
        # standard.trk contains only streamlines
        self.standard_trk_filename = os.path.join(DATA_PATH, "standard.trk")
        # standard.LPS.trk contains only streamlines
        self.standard_LPS_trk_filename = os.path.join(DATA_PATH, "standard.LPS.trk")

        # complex.trk contains streamlines, scalars and properties
        self.complex_trk_filename = os.path.join(DATA_PATH, "complex.trk")

        self.streamlines = [np.arange(1*3, dtype="f4").reshape((1, 3)),
                            np.arange(2*3, dtype="f4").reshape((2, 3)),
                            np.arange(5*3, dtype="f4").reshape((5, 3))]

        self.fa = [np.array([[0.2]], dtype="f4"),
                   np.array([[0.3],
                             [0.4]], dtype="f4"),
                   np.array([[0.5],
                             [0.6],
                             [0.6],
                             [0.7],
                             [0.8]], dtype="f4")]

        self.colors = [np.array([(1, 0, 0)]*1, dtype="f4"),
                       np.array([(0, 1, 0)]*2, dtype="f4"),
                       np.array([(0, 0, 1)]*5, dtype="f4")]

        self.mean_curvature = [np.array([1.11], dtype="f4"),
                               np.array([2.11], dtype="f4"),
                               np.array([3.11], dtype="f4")]

        self.mean_torsion = [np.array([1.22], dtype="f4"),
                             np.array([2.22], dtype="f4"),
                             np.array([3.22], dtype="f4")]

        self.mean_colors = [np.array([1, 0, 0], dtype="f4"),
                            np.array([0, 1, 0], dtype="f4"),
                            np.array([0, 0, 1], dtype="f4")]

        self.data_per_point = {'colors': self.colors,
                               'fa': self.fa}
        self.data_per_streamline = {'mean_curvature': self.mean_curvature,
                                    'mean_torsion': self.mean_torsion,
                                    'mean_colors': self.mean_colors}

        self.empty_tractogram = Tractogram()
        self.simple_tractogram = Tractogram(self.streamlines)
        self.complex_tractogram = Tractogram(self.streamlines,
                                             self.data_per_streamline,
                                             self.data_per_point)

    def test_load_empty_file(self):
        for lazy_load in [False, True]:
            trk = TrkFile.load(self.empty_trk_filename, lazy_load=lazy_load)
            assert_tractogram_equal(trk.tractogram, self.empty_tractogram)

    def test_load_simple_file(self):
        for lazy_load in [False, True]:
            trk = TrkFile.load(self.simple_trk_filename, lazy_load=lazy_load)
            assert_tractogram_equal(trk.tractogram, self.simple_tractogram)

    def test_load_complex_file(self):
        for lazy_load in [False, True]:
            trk = TrkFile.load(self.complex_trk_filename, lazy_load=lazy_load)
            assert_tractogram_equal(trk.tractogram, self.complex_tractogram)

    def test_load_file_with_wrong_information(self):
        trk_file = open(self.simple_trk_filename, 'rb').read()

        # Simulate a TRK file where `count` was not provided.
        count = np.array(0, dtype="int32").tostring()
        new_trk_file = trk_file[:1000-12] + count + trk_file[1000-8:]
        trk = TrkFile.load(BytesIO(new_trk_file), lazy_load=False)
        assert_tractogram_equal(trk.tractogram, self.simple_tractogram)

        # Simulate a TRK file where `voxel_order` was not provided.
        voxel_order = np.zeros(1, dtype="|S3").tostring()
        new_trk_file = trk_file[:948] + voxel_order + trk_file[948+3:]
        with clear_and_catch_warnings(record=True, modules=[trk_module]) as w:
            TrkFile.load(BytesIO(new_trk_file))
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

    def test_write_empty_file(self):
        tractogram = Tractogram()

        trk_file = BytesIO()
        trk = TrkFile(tractogram)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)

        new_trk = TrkFile.load(trk_file)
        assert_tractogram_equal(new_trk.tractogram, tractogram)

        new_trk_orig = TrkFile.load(self.empty_trk_filename)
        assert_tractogram_equal(new_trk.tractogram, new_trk_orig.tractogram)

        trk_file.seek(0, os.SEEK_SET)
        assert_equal(trk_file.read(), open(self.empty_trk_filename, 'rb').read())

    def test_write_simple_file(self):
        tractogram = Tractogram(self.streamlines)

        trk_file = BytesIO()
        trk = TrkFile(tractogram)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)

        new_trk = TrkFile.load(trk_file)
        assert_tractogram_equal(new_trk.tractogram, tractogram)

        new_trk_orig = TrkFile.load(self.simple_trk_filename)
        assert_tractogram_equal(new_trk.tractogram, new_trk_orig.tractogram)

        trk_file.seek(0, os.SEEK_SET)
        assert_equal(trk_file.read(), open(self.simple_trk_filename, 'rb').read())

    def test_write_complex_file(self):
        # With scalars
        tractogram = Tractogram(self.streamlines,
                                data_per_point=self.data_per_point)

        trk_file = BytesIO()
        trk = TrkFile(tractogram)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)

        new_trk = TrkFile.load(trk_file, lazy_load=False)
        assert_tractogram_equal(new_trk.tractogram, tractogram)

        # With properties
        tractogram = Tractogram(self.streamlines,
                                data_per_streamline=self.data_per_streamline)

        trk = TrkFile(tractogram)
        trk_file = BytesIO()
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)

        new_trk = TrkFile.load(trk_file, lazy_load=False)
        assert_tractogram_equal(new_trk.tractogram, tractogram)

        # With scalars and properties
        tractogram = Tractogram(self.streamlines,
                                data_per_point=self.data_per_point,
                                data_per_streamline=self.data_per_streamline)

        trk_file = BytesIO()
        trk = TrkFile(tractogram)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)

        new_trk = TrkFile.load(trk_file, lazy_load=False)
        assert_tractogram_equal(new_trk.tractogram, tractogram)

        new_trk_orig = TrkFile.load(self.complex_trk_filename)
        assert_tractogram_equal(new_trk.tractogram, new_trk_orig.tractogram)

        trk_file.seek(0, os.SEEK_SET)
        assert_equal(trk_file.read(),
                     open(self.complex_trk_filename, 'rb').read())

    def test_load_write_file(self):
        for filename in [self.empty_trk_filename, self.simple_trk_filename, self.complex_trk_filename]:
            for lazy_load in [False, True]:
                trk = TrkFile.load(filename, lazy_load=lazy_load)
                trk_file = BytesIO()
                trk.save(trk_file)

                new_trk = TrkFile.load(filename, lazy_load=False)
                assert_tractogram_equal(new_trk.tractogram, trk.tractogram)

                trk_file.seek(0, os.SEEK_SET)
                #assert_equal(trk_file.read(), open(filename, 'rb').read())

    def test_load_write_LPS_file(self):
        # Load the RAS and LPS version of the standard.
        trk_RAS = TrkFile.load(self.standard_trk_filename, lazy_load=False)
        trk_LPS = TrkFile.load(self.standard_LPS_trk_filename, lazy_load=False)
        assert_tractogram_equal(trk_LPS.tractogram, trk_RAS.tractogram)

        # Write back the standard.
        trk_file = BytesIO()
        trk = TrkFile(trk_LPS.tractogram, trk_LPS.header)
        trk.save(trk_file)
        trk_file.seek(0, os.SEEK_SET)

        new_trk = TrkFile.load(trk_file)

        assert_header_equal(new_trk.header, trk.header)
        assert_tractogram_equal(new_trk.tractogram, trk.tractogram)

        new_trk_orig = TrkFile.load(self.standard_LPS_trk_filename)
        assert_tractogram_equal(new_trk.tractogram, new_trk_orig.tractogram)

        trk_file.seek(0, os.SEEK_SET)
        assert_equal(trk_file.read(),
                     open(self.standard_LPS_trk_filename, 'rb').read())

    def test_write_too_many_scalars_and_properties(self):
        # TRK supports up to 10 data_per_point.
        data_per_point = {}
        for i in range(10):
            data_per_point['#{0}'.format(i)] = self.fa

            tractogram = Tractogram(self.streamlines,
                                    data_per_point=data_per_point)

            trk_file = BytesIO()
            trk = TrkFile(tractogram)
            trk.save(trk_file)
            trk_file.seek(0, os.SEEK_SET)

            new_trk = TrkFile.load(trk_file, lazy_load=False)
            assert_tractogram_equal(new_trk.tractogram, tractogram)

        # More than 10 data_per_point should raise an error.
        data_per_point['#{0}'.format(i+1)] = self.fa

        tractogram = Tractogram(self.streamlines,
                                data_per_point=data_per_point)

        trk = TrkFile(tractogram)
        assert_raises(ValueError, trk.save, BytesIO())

        # TRK supports up to 10 data_per_streamline.
        data_per_streamline = {}
        for i in range(10):
            data_per_streamline['#{0}'.format(i)] = self.mean_torsion

            tractogram = Tractogram(self.streamlines,
                                    data_per_streamline=data_per_streamline)

            trk_file = BytesIO()
            trk = TrkFile(tractogram)
            trk.save(trk_file)
            trk_file.seek(0, os.SEEK_SET)

            new_trk = TrkFile.load(trk_file, lazy_load=False)
            assert_tractogram_equal(new_trk.tractogram, tractogram)

        # More than 10 data_per_streamline should raise an error.
        data_per_streamline['#{0}'.format(i+1)] = self.mean_torsion

        tractogram = Tractogram(self.streamlines,
                                data_per_streamline=data_per_streamline)

        trk = TrkFile(tractogram)
        assert_raises(ValueError, trk.save, BytesIO())

    def test_write_scalars_and_properties_name_too_long(self):
        # TRK supports data_per_point name up to 20 characters.
        # However, we reserve the last two characters to store
        # the number of values associated to each data_per_point.
        # So in reality we allow name of 18 characters, otherwise
        # the name is truncated and warning is issue.
        for nb_chars in range(22):
            data_per_point = {'A'*nb_chars: self.colors}
            tractogram = Tractogram(self.streamlines,
                                    data_per_point=data_per_point)

            trk = TrkFile(tractogram)
            if nb_chars > 18:
                assert_raises(ValueError, trk.save, BytesIO())
            else:
                trk.save(BytesIO())

            data_per_point = {'A'*nb_chars: self.fa}
            tractogram = Tractogram(self.streamlines,
                                    data_per_point=data_per_point)

            trk = TrkFile(tractogram)
            if nb_chars > 20:
                assert_raises(ValueError, trk.save, BytesIO())
            else:
                trk.save(BytesIO())

        # TRK supports data_per_streamline name up to 20 characters.
        # However, we reserve the last two characters to store
        # the number of values associated to each data_per_streamline.
        # So in reality we allow name of 18 characters, otherwise
        # the name is truncated and warning is issue.
        for nb_chars in range(22):
            data_per_streamline = {'A'*nb_chars: self.mean_colors}
            tractogram = Tractogram(self.streamlines,
                                    data_per_streamline=data_per_streamline)

            trk = TrkFile(tractogram)
            if nb_chars > 18:
                assert_raises(ValueError, trk.save, BytesIO())
            else:
                trk.save(BytesIO())

            data_per_streamline = {'A'*nb_chars: self.mean_torsion}
            tractogram = Tractogram(self.streamlines,
                                    data_per_streamline=data_per_streamline)

            trk = TrkFile(tractogram)
            if nb_chars > 20:
                assert_raises(ValueError, trk.save, BytesIO())
            else:
                trk.save(BytesIO())
