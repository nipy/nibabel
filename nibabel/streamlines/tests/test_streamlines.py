import os
import unittest
import tempfile
import numpy as np

from os.path import join as pjoin

import nibabel as nib
from nibabel.externals.six import BytesIO

from nibabel.testing import clear_and_catch_warnings
from nibabel.testing import assert_arrays_equal, check_iteration
from nose.tools import assert_equal, assert_raises, assert_true, assert_false

from .test_tractogram import assert_tractogram_equal
from ..tractogram import Tractogram, LazyTractogram
from ..tractogram_file import TractogramFile
from ..tractogram import UsageWarning
from .. import trk

DATA_PATH = pjoin(os.path.dirname(__file__), 'data')


def test_is_supported():
    # Emtpy file/string
    f = BytesIO()
    assert_false(nib.streamlines.is_supported(f))
    assert_false(nib.streamlines.is_supported(""))

    # Valid file without extension
    for tractogram_file in nib.streamlines.FORMATS.values():
        f = BytesIO()
        f.write(tractogram_file.get_magic_number())
        f.seek(0, os.SEEK_SET)
        assert_true(nib.streamlines.is_supported(f))

    # Wrong extension but right magic number
    for tractogram_file in nib.streamlines.FORMATS.values():
        with tempfile.TemporaryFile(mode="w+b", suffix=".txt") as f:
            f.write(tractogram_file.get_magic_number())
            f.seek(0, os.SEEK_SET)
            assert_true(nib.streamlines.is_supported(f))

    # Good extension but wrong magic number
    for ext, tractogram_file in nib.streamlines.FORMATS.items():
        with tempfile.TemporaryFile(mode="w+b", suffix=ext) as f:
            f.write(b"pass")
            f.seek(0, os.SEEK_SET)
            assert_false(nib.streamlines.is_supported(f))

    # Wrong extension, string only
    f = "my_tractogram.asd"
    assert_false(nib.streamlines.is_supported(f))

    # Good extension, string only
    for ext, tractogram_file in nib.streamlines.FORMATS.items():
        f = "my_tractogram" + ext
        assert_true(nib.streamlines.is_supported(f))


def test_detect_format():
    # Emtpy file/string
    f = BytesIO()
    assert_equal(nib.streamlines.detect_format(f), None)
    assert_equal(nib.streamlines.detect_format(""), None)

    # Valid file without extension
    for tractogram_file in nib.streamlines.FORMATS.values():
        f = BytesIO()
        f.write(tractogram_file.get_magic_number())
        f.seek(0, os.SEEK_SET)
        assert_equal(nib.streamlines.detect_format(f), tractogram_file)

    # Wrong extension but right magic number
    for tractogram_file in nib.streamlines.FORMATS.values():
        with tempfile.TemporaryFile(mode="w+b", suffix=".txt") as f:
            f.write(tractogram_file.get_magic_number())
            f.seek(0, os.SEEK_SET)
            assert_equal(nib.streamlines.detect_format(f), tractogram_file)

    # Good extension but wrong magic number
    for ext, tractogram_file in nib.streamlines.FORMATS.items():
        with tempfile.TemporaryFile(mode="w+b", suffix=ext) as f:
            f.write(b"pass")
            f.seek(0, os.SEEK_SET)
            assert_equal(nib.streamlines.detect_format(f), None)

    # Wrong extension, string only
    f = "my_tractogram.asd"
    assert_equal(nib.streamlines.detect_format(f), None)

    # Good extension, string only
    for ext, tractogram_file in nib.streamlines.FORMATS.items():
        f = "my_tractogram" + ext
        assert_equal(nib.streamlines.detect_format(f), tractogram_file)


class TestLoadSave(unittest.TestCase):
    def setUp(self):
        self.empty_filenames = [pjoin(DATA_PATH, "empty" + ext)
                                for ext in nib.streamlines.FORMATS.keys()]
        self.simple_filenames = [pjoin(DATA_PATH, "simple" + ext)
                                 for ext in nib.streamlines.FORMATS.keys()]
        self.complex_filenames = [pjoin(DATA_PATH, "complex" + ext)
                                  for ext in nib.streamlines.FORMATS.keys()]

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
            for empty_filename in self.empty_filenames:
                tfile = nib.streamlines.load(empty_filename,
                                             lazy_load=lazy_load)
                assert_true(isinstance(tfile, TractogramFile))

                if lazy_load:
                    assert_true(type(tfile.tractogram), Tractogram)
                else:
                    assert_true(type(tfile.tractogram), LazyTractogram)

                assert_tractogram_equal(tfile.tractogram,
                                        self.empty_tractogram)

    def test_load_simple_file(self):
        for lazy_load in [False, True]:
            for simple_filename in self.simple_filenames:
                tfile = nib.streamlines.load(simple_filename,
                                             lazy_load=lazy_load)
                assert_true(isinstance(tfile, TractogramFile))

                if lazy_load:
                    assert_true(type(tfile.tractogram), Tractogram)
                else:
                    assert_true(type(tfile.tractogram), LazyTractogram)

                assert_tractogram_equal(tfile.tractogram,
                                        self.simple_tractogram)

    def test_load_complex_file(self):
        for lazy_load in [False, True]:
            for complex_filename in self.complex_filenames:
                tfile = nib.streamlines.load(complex_filename,
                                             lazy_load=lazy_load)
                assert_true(isinstance(tfile, TractogramFile))

                if lazy_load:
                    assert_true(type(tfile.tractogram), Tractogram)
                else:
                    assert_true(type(tfile.tractogram), LazyTractogram)

                tractogram = Tractogram(self.streamlines)

                if tfile.support_data_per_point():
                    tractogram.data_per_point = self.data_per_point

                if tfile.support_data_per_streamline():
                    tractogram.data_per_streamline = self.data_per_streamline

                assert_tractogram_equal(tfile.tractogram,
                                        tractogram)

    def test_save_empty_file(self):
        tractogram = Tractogram()
        for ext, cls in nib.streamlines.FORMATS.items():
            with tempfile.NamedTemporaryFile(mode="w+b", suffix=ext) as f:
                nib.streamlines.save_tractogram(tractogram, f.name)
                tfile = nib.streamlines.load(f, lazy_load=False)
                assert_tractogram_equal(tfile.tractogram, tractogram)

    def test_save_simple_file(self):
        tractogram = Tractogram(self.streamlines)
        for ext, cls in nib.streamlines.FORMATS.items():
            with tempfile.NamedTemporaryFile(mode="w+b", suffix=ext) as f:
                nib.streamlines.save_tractogram(tractogram, f.name)
                tfile = nib.streamlines.load(f, lazy_load=False)
                assert_tractogram_equal(tfile.tractogram, tractogram)

    def test_save_complex_file(self):
        complex_tractogram = Tractogram(self.streamlines,
                                        self.data_per_streamline,
                                        self.data_per_point)

        for ext, cls in nib.streamlines.FORMATS.items():
            with tempfile.NamedTemporaryFile(mode="w+b", suffix=ext) as f:
                with clear_and_catch_warnings(record=True, modules=[trk]) as w:
                    nib.streamlines.save_tractogram(complex_tractogram, f.name)

                    # If streamlines format does not support saving data per
                    # point or data per streamline, a warning message should
                    # be issued.
                    if not (cls.support_data_per_point()
                            and cls.support_data_per_streamline()):
                        assert_equal(len(w), 1)
                        assert_true(issubclass(w[0].category, UsageWarning))

                tractogram = Tractogram(self.streamlines)

                if cls.support_data_per_point():
                    tractogram.data_per_point = self.data_per_point

                if cls.support_data_per_streamline():
                    tractogram.data_per_streamline = self.data_per_streamline

                tfile = nib.streamlines.load(f, lazy_load=False)
                assert_tractogram_equal(tfile.tractogram, tractogram)
