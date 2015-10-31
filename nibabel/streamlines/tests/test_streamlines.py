import os
import unittest
import tempfile
import numpy as np

from os.path import join as pjoin

import nibabel as nib
from nibabel.externals.six import BytesIO

from nibabel.testing import clear_and_catch_warnings
from nibabel.testing import assert_arrays_equal
from nose.tools import assert_equal, assert_raises, assert_true, assert_false

from ..base_format import Tractogram, LazyTractogram
from ..base_format import HeaderError, UsageWarning
from ..header import Field
from .. import trk

DATA_PATH = pjoin(os.path.dirname(__file__), 'data')


def isiterable(streamlines):
    try:
        for _ in streamlines:
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


def test_is_supported():
    # Emtpy file/string
    f = BytesIO()
    assert_false(nib.streamlines.is_supported(f))
    assert_false(nib.streamlines.is_supported(""))

    # Valid file without extension
    for streamlines_file in nib.streamlines.FORMATS.values():
        f = BytesIO()
        f.write(streamlines_file.get_magic_number())
        f.seek(0, os.SEEK_SET)
        assert_true(nib.streamlines.is_supported(f))

    # Wrong extension but right magic number
    for streamlines_file in nib.streamlines.FORMATS.values():
        with tempfile.TemporaryFile(mode="w+b", suffix=".txt") as f:
            f.write(streamlines_file.get_magic_number())
            f.seek(0, os.SEEK_SET)
            assert_true(nib.streamlines.is_supported(f))

    # Good extension but wrong magic number
    for ext, streamlines_file in nib.streamlines.FORMATS.items():
        with tempfile.TemporaryFile(mode="w+b", suffix=ext) as f:
            f.write(b"pass")
            f.seek(0, os.SEEK_SET)
            assert_false(nib.streamlines.is_supported(f))

    # Wrong extension, string only
    f = "my_streamlines.asd"
    assert_false(nib.streamlines.is_supported(f))

    # Good extension, string only
    for ext, streamlines_file in nib.streamlines.FORMATS.items():
        f = "my_streamlines" + ext
        assert_true(nib.streamlines.is_supported(f))


def test_detect_format():
    # Emtpy file/string
    f = BytesIO()
    assert_equal(nib.streamlines.detect_format(f), None)
    assert_equal(nib.streamlines.detect_format(""), None)

    # Valid file without extension
    for streamlines_file in nib.streamlines.FORMATS.values():
        f = BytesIO()
        f.write(streamlines_file.get_magic_number())
        f.seek(0, os.SEEK_SET)
        assert_equal(nib.streamlines.detect_format(f), streamlines_file)

    # Wrong extension but right magic number
    for streamlines_file in nib.streamlines.FORMATS.values():
        with tempfile.TemporaryFile(mode="w+b", suffix=".txt") as f:
            f.write(streamlines_file.get_magic_number())
            f.seek(0, os.SEEK_SET)
            assert_equal(nib.streamlines.detect_format(f), streamlines_file)

    # Good extension but wrong magic number
    for ext, streamlines_file in nib.streamlines.FORMATS.items():
        with tempfile.TemporaryFile(mode="w+b", suffix=ext) as f:
            f.write(b"pass")
            f.seek(0, os.SEEK_SET)
            assert_equal(nib.streamlines.detect_format(f), None)

    # Wrong extension, string only
    f = "my_streamlines.asd"
    assert_equal(nib.streamlines.detect_format(f), None)

    # Good extension, string only
    for ext, streamlines_file in nib.streamlines.FORMATS.items():
        f = "my_streamlines" + ext
        assert_equal(nib.streamlines.detect_format(f), streamlines_file)


class TestLoadSave(unittest.TestCase):
    def setUp(self):
        self.empty_filenames = [pjoin(DATA_PATH, "empty" + ext) for ext in nib.streamlines.FORMATS.keys()]
        self.simple_filenames = [pjoin(DATA_PATH, "simple" + ext) for ext in nib.streamlines.FORMATS.keys()]
        self.complex_filenames = [pjoin(DATA_PATH, "complex" + ext) for ext in nib.streamlines.FORMATS.keys()]

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
        self.to_world_space = np.eye(4)

    def test_load_empty_file(self):
        for empty_filename in self.empty_filenames:
            streamlines = nib.streamlines.load(empty_filename,
                                               ref=self.to_world_space,
                                               lazy_load=False)
            assert_true(type(streamlines), Tractogram)
            check_streamlines(streamlines, 0, [], [], [])

    def test_load_simple_file(self):
        for simple_filename in self.simple_filenames:
            streamlines = nib.streamlines.load(simple_filename,
                                               ref=self.to_world_space,
                                               lazy_load=False)
            assert_true(type(streamlines), Tractogram)
            check_streamlines(streamlines, self.nb_streamlines,
                              self.points, [], [])

            # Test lazy_load
            streamlines = nib.streamlines.load(simple_filename,
                                               ref=self.to_world_space,
                                               lazy_load=True)
            assert_true(type(streamlines), LazyTractogram)
            check_streamlines(streamlines, self.nb_streamlines,
                              self.points, [], [])

    def test_load_complex_file(self):
        for complex_filename in self.complex_filenames:
            file_format = nib.streamlines.detect_format(complex_filename)

            scalars = []
            if file_format.can_save_scalars():
                scalars = self.colors

            properties = []
            if file_format.can_save_properties():
                properties = self.mean_curvature_torsion

            streamlines = nib.streamlines.load(complex_filename,
                                               ref=self.to_world_space,
                                               lazy_load=False)
            assert_true(type(streamlines), Tractogram)
            check_streamlines(streamlines, self.nb_streamlines,
                              self.points, scalars, properties)

            # Test lazy_load
            streamlines = nib.streamlines.load(complex_filename,
                                               ref=self.to_world_space,
                                               lazy_load=True)
            assert_true(type(streamlines), LazyTractogram)
            check_streamlines(streamlines, self.nb_streamlines,
                              self.points, scalars, properties)

    def test_save_simple_file(self):
        streamlines = Tractogram(self.points)
        for ext in nib.streamlines.FORMATS.keys():
            with tempfile.NamedTemporaryFile(mode="w+b", suffix=ext) as f:
                nib.streamlines.save(streamlines, f.name)
                loaded_streamlines = nib.streamlines.load(f, ref=self.to_world_space, lazy_load=False)
                check_streamlines(loaded_streamlines, self.nb_streamlines,
                                  self.points, [], [])

    def test_save_complex_file(self):
        streamlines = Tractogram(self.points, scalars=self.colors, properties=self.mean_curvature_torsion)
        for ext, cls in nib.streamlines.FORMATS.items():
            with tempfile.NamedTemporaryFile(mode="w+b", suffix=ext) as f:
                with clear_and_catch_warnings(record=True, modules=[trk]) as w:
                    nib.streamlines.save(streamlines, f.name)

                    # If streamlines format does not support saving scalars or
                    # properties, a warning message should be issued.
                    if not (cls.can_save_scalars() and cls.can_save_properties()):
                        assert_equal(len(w), 1)
                        assert_true(issubclass(w[0].category, UsageWarning))

                scalars = []
                if cls.can_save_scalars():
                    scalars = self.colors

                properties = []
                if cls.can_save_properties():
                    properties = self.mean_curvature_torsion

                loaded_streamlines = nib.streamlines.load(f, ref=self.to_world_space, lazy_load=False)
                check_streamlines(loaded_streamlines, self.nb_streamlines,
                                  self.points, scalars, properties)
