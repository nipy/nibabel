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

from ..tck import TckFile

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


class TestTCK(unittest.TestCase):

    def setUp(self):
        self.empty_tck_filename = os.path.join(DATA_PATH, "empty.tck")
        # simple.tck contains only streamlines
        self.simple_tck_filename = os.path.join(DATA_PATH, "simple.tck")

        self.streamlines = [np.arange(1*3, dtype="f4").reshape((1, 3)),
                            np.arange(2*3, dtype="f4").reshape((2, 3)),
                            np.arange(5*3, dtype="f4").reshape((5, 3))]

        self.empty_tractogram = Tractogram()
        self.simple_tractogram = Tractogram(self.streamlines)

    def test_load_empty_file(self):
        for lazy_load in [False, True]:
            tck = TckFile.load(self.empty_tck_filename, lazy_load=lazy_load)
            assert_tractogram_equal(tck.tractogram, self.empty_tractogram)

    def test_load_simple_file(self):
        for lazy_load in [False, True]:
            tck = TckFile.load(self.simple_tck_filename, lazy_load=lazy_load)
            assert_tractogram_equal(tck.tractogram, self.simple_tractogram)

    def test_write_empty_file(self):
        tractogram = Tractogram()

        tck_file = BytesIO()
        tck = TckFile(tractogram)
        tck.save(tck_file)
        tck_file.seek(0, os.SEEK_SET)

        loaded_tck = TckFile.load(tck_file)
        assert_tractogram_equal(loaded_tck.tractogram, tractogram)

        loaded_tck_orig = TckFile.load(self.empty_tck_filename)
        assert_tractogram_equal(loaded_tck.tractogram, loaded_tck_orig.tractogram)

        tck_file.seek(0, os.SEEK_SET)
        assert_equal(tck_file.read(), open(self.empty_tck_filename, 'rb').read())

    def test_write_simple_file(self):
        tractogram = Tractogram(self.streamlines)

        tck_file = BytesIO()
        tck = TckFile(tractogram)
        tck.save(tck_file)
        tck_file.seek(0, os.SEEK_SET)

        loaded_tck = TckFile.load(tck_file)
        assert_tractogram_equal(loaded_tck.tractogram, tractogram)

        loaded_tck_orig = TckFile.load(self.simple_tck_filename)
        assert_tractogram_equal(loaded_tck.tractogram, loaded_tck_orig.tractogram)

        tck_file.seek(0, os.SEEK_SET)
        assert_equal(tck_file.read(), open(self.simple_tck_filename, 'rb').read())

    def test_load_write_file(self):
        for filename in [self.empty_tck_filename, self.simple_tck_filename]:
            for lazy_load in [False, True]:
                tck = TckFile.load(filename, lazy_load=lazy_load)
                tck_file = BytesIO()
                tck.save(tck_file)

                loaded_tck = TckFile.load(filename, lazy_load=False)
                assert_tractogram_equal(loaded_tck.tractogram, tck.tractogram)

                tck_file.seek(0, os.SEEK_SET)
                #assert_equal(tck_file.read(), open(filename, 'rb').read())
