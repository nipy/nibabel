import os
import unittest
import tempfile
import numpy as np

from os.path import join as pjoin

import nibabel as nib
from nibabel.externals.six import BytesIO

from nibabel.testing import clear_and_catch_warnings
from nibabel.testing import assert_arrays_equal, isiterable
from nose.tools import assert_equal, assert_raises, assert_true, assert_false

from ..base_format import Tractogram, LazyTractogram, TractogramFile
from ..base_format import HeaderError, UsageWarning
from ..header import Field
from .. import trk

DATA_PATH = pjoin(os.path.dirname(__file__), 'data')


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
        self.empty_filenames = [pjoin(DATA_PATH, "empty" + ext) for ext in nib.streamlines.FORMATS.keys()]
        self.simple_filenames = [pjoin(DATA_PATH, "simple" + ext) for ext in nib.streamlines.FORMATS.keys()]
        self.complex_filenames = [pjoin(DATA_PATH, "complex" + ext) for ext in nib.streamlines.FORMATS.keys()]

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
        self.to_world_space = np.eye(4)

    def test_load_empty_file(self):
        for empty_filename in self.empty_filenames:
            tractogram_file = nib.streamlines.load(empty_filename,
                                                   lazy_load=False,
                                                   ref=self.to_world_space)
            assert_true(isinstance(tractogram_file, TractogramFile))
            assert_true(type(tractogram_file.tractogram), Tractogram)
            check_tractogram(tractogram_file.tractogram, 0, [], [], [])

    def test_load_simple_file(self):
        for simple_filename in self.simple_filenames:
            tractogram_file = nib.streamlines.load(simple_filename,
                                                   lazy_load=False,
                                                   ref=self.to_world_space)
            assert_true(isinstance(tractogram_file, TractogramFile))
            assert_true(type(tractogram_file.tractogram), Tractogram)
            check_tractogram(tractogram_file.tractogram, self.nb_streamlines,
                             self.streamlines, [], [])

            # Test lazy_load
            tractogram_file = nib.streamlines.load(simple_filename,
                                                   lazy_load=True,
                                                   ref=self.to_world_space)

            assert_true(isinstance(tractogram_file, TractogramFile))
            assert_true(type(tractogram_file.tractogram), LazyTractogram)
            check_tractogram(tractogram_file.tractogram, self.nb_streamlines,
                             self.streamlines, [], [])

    def test_load_complex_file(self):
        for complex_filename in self.complex_filenames:
            file_format = nib.streamlines.detect_format(complex_filename)

            scalars = []
            if file_format.can_save_scalars():
                scalars = self.colors

            properties = []
            if file_format.can_save_properties():
                properties = self.mean_curvature_torsion

            tractogram_file = nib.streamlines.load(complex_filename,
                                                   lazy_load=False,
                                                   ref=self.to_world_space)
            assert_true(isinstance(tractogram_file, TractogramFile))
            assert_true(type(tractogram_file.tractogram), Tractogram)
            check_tractogram(tractogram_file.tractogram, self.nb_streamlines,
                             self.streamlines, scalars, properties)

            # Test lazy_load
            tractogram_file = nib.streamlines.load(complex_filename,
                                                   lazy_load=True,
                                                   ref=self.to_world_space)
            assert_true(isinstance(tractogram_file, TractogramFile))
            assert_true(type(tractogram_file.tractogram), LazyTractogram)
            check_tractogram(tractogram_file.tractogram, self.nb_streamlines,
                             self.streamlines, scalars, properties)

    def test_save_simple_file(self):
        tractogram = Tractogram(self.streamlines)
        for ext in nib.streamlines.FORMATS.keys():
            with tempfile.NamedTemporaryFile(mode="w+b", suffix=ext) as f:
                nib.streamlines.save(tractogram, f.name)
                loaded_tractogram = nib.streamlines.load(f, ref=self.to_world_space, lazy_load=False)
                check_tractogram(loaded_tractogram, self.nb_streamlines,
                                  self.streamlines, [], [])

    def test_save_complex_file(self):
        tractogram = Tractogram(self.streamlines, scalars=self.colors, properties=self.mean_curvature_torsion)
        for ext, cls in nib.streamlines.FORMATS.items():
            with tempfile.NamedTemporaryFile(mode="w+b", suffix=ext) as f:
                with clear_and_catch_warnings(record=True, modules=[trk]) as w:
                    nib.streamlines.save(tractogram, f.name)

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

                loaded_tractogram = nib.streamlines.load(f, ref=self.to_world_space, lazy_load=False)
                check_tractogram(loaded_tractogram, self.nb_streamlines,
                                  self.streamlines, scalars, properties)
