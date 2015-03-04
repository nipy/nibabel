import os
import unittest
import tempfile
import numpy as np

from os.path import join as pjoin

from nibabel.externals.six import BytesIO

from nibabel.testing import assert_arrays_equal
from nose.tools import assert_equal, assert_raises, assert_true, assert_false

import nibabel.streamlines.utils as streamline_utils

from nibabel.streamlines.base_format import Streamlines, LazyStreamlines
from nibabel.streamlines.base_format import HeaderError
from nibabel.streamlines.header import Field

DATA_PATH = pjoin(os.path.dirname(__file__), 'data')


def test_is_supported():
    # Emtpy file/string
    f = BytesIO()
    assert_false(streamline_utils.is_supported(f))
    assert_false(streamline_utils.is_supported(""))

    # Valid file without extension
    for streamlines_file in streamline_utils.FORMATS.values():
        f = BytesIO()
        f.write(streamlines_file.get_magic_number())
        f.seek(0, os.SEEK_SET)
        assert_true(streamline_utils.is_supported(f))

    # Wrong extension but right magic number
    for streamlines_file in streamline_utils.FORMATS.values():
        with tempfile.TemporaryFile(mode="w+b", suffix=".txt") as f:
            f.write(streamlines_file.get_magic_number())
            f.seek(0, os.SEEK_SET)
            assert_true(streamline_utils.is_supported(f))

    # Good extension but wrong magic number
    for ext, streamlines_file in streamline_utils.FORMATS.items():
        with tempfile.TemporaryFile(mode="w+b", suffix=ext) as f:
            f.write(b"pass")
            f.seek(0, os.SEEK_SET)
            assert_false(streamline_utils.is_supported(f))

    # Wrong extension, string only
    f = "my_streamlines.asd"
    assert_false(streamline_utils.is_supported(f))

    # Good extension, string only
    for ext, streamlines_file in streamline_utils.FORMATS.items():
        f = "my_streamlines" + ext
        assert_true(streamline_utils.is_supported(f))


def test_detect_format():
    # Emtpy file/string
    f = BytesIO()
    assert_equal(streamline_utils.detect_format(f), None)
    assert_equal(streamline_utils.detect_format(""), None)

    # Valid file without extension
    for streamlines_file in streamline_utils.FORMATS.values():
        f = BytesIO()
        f.write(streamlines_file.get_magic_number())
        f.seek(0, os.SEEK_SET)
        assert_equal(streamline_utils.detect_format(f), streamlines_file)

    # Wrong extension but right magic number
    for streamlines_file in streamline_utils.FORMATS.values():
        with tempfile.TemporaryFile(mode="w+b", suffix=".txt") as f:
            f.write(streamlines_file.get_magic_number())
            f.seek(0, os.SEEK_SET)
            assert_equal(streamline_utils.detect_format(f), streamlines_file)

    # Good extension but wrong magic number
    for ext, streamlines_file in streamline_utils.FORMATS.items():
        with tempfile.TemporaryFile(mode="w+b", suffix=ext) as f:
            f.write(b"pass")
            f.seek(0, os.SEEK_SET)
            assert_equal(streamline_utils.detect_format(f), None)

    # Wrong extension, string only
    f = "my_streamlines.asd"
    assert_equal(streamline_utils.detect_format(f), None)

    # Good extension, string only
    for ext, streamlines_file in streamline_utils.FORMATS.items():
        f = "my_streamlines" + ext
        assert_equal(streamline_utils.detect_format(f), streamlines_file)


class TestLoadSave(unittest.TestCase):
    # Testing scalars and properties depend on the format.
    # See unit tests in the specific format test file.

    def setUp(self):
        self.empty_filenames = [pjoin(DATA_PATH, "empty" + ext) for ext in streamline_utils.FORMATS.keys()]
        self.simple_filenames = [pjoin(DATA_PATH, "simple" + ext) for ext in streamline_utils.FORMATS.keys()]
        self.complex_filenames = [pjoin(DATA_PATH, "complex" + ext) for ext in streamline_utils.FORMATS.keys()]

        self.points = [np.arange(1*3, dtype="f4").reshape((1, 3)),
                       np.arange(2*3, dtype="f4").reshape((2, 3)),
                       np.arange(5*3, dtype="f4").reshape((5, 3))]

        self.colors = [np.array([(1, 0, 0)]*1, dtype="f4"),
                       np.array([(0, 1, 0)]*2, dtype="f4"),
                       np.array([(0, 0, 1)]*5, dtype="f4")]

        self.mean_curvature_torsion = [np.array([1.11, 1.22], dtype="f4"),
                                       np.array([2.11, 2.22], dtype="f4"),
                                       np.array([3.11, 3.22], dtype="f4")]

    def test_load_empty_file(self):
        for empty_filename in self.empty_filenames:
            empty_streamlines = streamline_utils.load(empty_filename, None, lazy_load=False)

            hdr = empty_streamlines.header
            assert_equal(hdr[Field.NB_STREAMLINES], 0)
            assert_equal(len(empty_streamlines), 0)

            points = empty_streamlines.points
            assert_equal(len(points), 0)

            # For an empty file, scalars should be zero regardless of the format.
            scalars = empty_streamlines.scalars
            assert_equal(len(scalars), 0)

            # For an empty file, properties should be zero regardless of the format.
            properties = empty_streamlines.properties
            assert_equal(len(properties), 0)

            # Check if we can iterate through the streamlines.
            for point, scalar, prop in empty_streamlines:
                pass

    def test_load_simple_file(self):
        for simple_filename in self.simple_filenames:
            simple_streamlines = streamline_utils.load(simple_filename, None, lazy_load=False)
            assert_true(type(simple_streamlines), Streamlines)

            hdr = simple_streamlines.header
            assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))
            assert_equal(len(simple_streamlines), len(self.points))

            points = simple_streamlines.points
            assert_arrays_equal(points, self.points)

            # Testing scalars and properties depend on the format.
            # See unit tests in the specific format test file.

            # Check if we can iterate through the streamlines.
            for point, scalar, prop in simple_streamlines:
                pass

            # Test lazy_load
            simple_streamlines = streamline_utils.load(simple_filename, None, lazy_load=True)
            assert_true(type(simple_streamlines), LazyStreamlines)

            hdr = simple_streamlines.header
            assert_true(Field.NB_STREAMLINES in hdr)
            assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))

            points = simple_streamlines.points
            assert_arrays_equal(points, self.points)

            # Check if we can iterate through the streamlines.
            for point, scalar, prop in simple_streamlines:
                pass

    def test_load_complex_file(self):
        for complex_filename in self.complex_filenames:
            complex_streamlines = streamline_utils.load(complex_filename, None, lazy_load=False)

        hdr = complex_streamlines.header
        assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))
        assert_equal(len(complex_streamlines), len(self.points))

        points = complex_streamlines.points
        assert_arrays_equal(points, self.points)

        # Check if we can iterate through the streamlines.
        for point, scalar, prop in complex_streamlines:
            pass

        complex_streamlines = streamline_utils.load(complex_filename, None, lazy_load=True)

        hdr = complex_streamlines.header
        assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))
        assert_equal(len(complex_streamlines), len(self.points))

        points = complex_streamlines.points
        assert_arrays_equal(points, self.points)

        # Check if we can iterate through the streamlines.
        for point, scalar, prop in complex_streamlines:
            pass

    def test_save_simple_file(self):
        for ext in streamline_utils.FORMATS.keys():
            with tempfile.NamedTemporaryFile(mode="w+b", suffix=ext) as f:
                streamlines = Streamlines(self.points)

                streamline_utils.save(streamlines, f.name)
                simple_streamlines = streamline_utils.load(f, None, lazy_load=False)

                hdr = simple_streamlines.header
                assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))
                assert_equal(len(simple_streamlines), len(self.points))

                points = simple_streamlines.points
                assert_arrays_equal(points, self.points)

    def test_save_complex_file(self):
        for ext in streamline_utils.FORMATS.keys():
            # With scalars
            with tempfile.NamedTemporaryFile(mode="w+b", suffix=ext) as f:
                streamlines = Streamlines(self.points, scalars=self.colors)

                streamline_utils.save(streamlines, f.name)
                complex_streamlines = streamline_utils.load(f, None, lazy_load=False)

                hdr = complex_streamlines.header
                assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))
                assert_equal(len(complex_streamlines), len(self.points))

                points = complex_streamlines.points
                assert_arrays_equal(points, self.points)

                # Check if we can iterate through the streamlines.
                for point, scalar, prop in complex_streamlines:
                    pass

            # With properties
            with tempfile.NamedTemporaryFile(mode="w+b", suffix=ext) as f:
                streamlines = Streamlines(self.points, properties=self.mean_curvature_torsion)

                streamline_utils.save(streamlines, f.name)
                complex_streamlines = streamline_utils.load(f, None, lazy_load=False)

                hdr = complex_streamlines.header
                assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))
                assert_equal(len(complex_streamlines), len(self.points))

                points = complex_streamlines.points
                assert_arrays_equal(points, self.points)

                # Check if we can iterate through the streamlines.
                for point, scalar, prop in complex_streamlines:
                    pass

            # With scalars and properties
            with tempfile.NamedTemporaryFile(mode="w+b", suffix=ext) as f:
                streamlines = Streamlines(self.points, scalars=self.colors, properties=self.mean_curvature_torsion)

                streamline_utils.save(streamlines, f.name)
                complex_streamlines = streamline_utils.load(f, None, lazy_load=False)

                hdr = complex_streamlines.header
                assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))
                assert_equal(len(complex_streamlines), len(self.points))

                points = complex_streamlines.points
                assert_arrays_equal(points, self.points)

                # Check if we can iterate through the streamlines.
                for point, scalar, prop in complex_streamlines:
                    pass

    def test_save_file_from_generator(self):
        for ext in streamline_utils.FORMATS.keys():
            # With scalars
            with tempfile.NamedTemporaryFile(mode="w+b", suffix=ext) as f:
                gen_points = (point for point in self.points)
                gen_scalars = (scalar for scalar in self.colors)
                gen_properties = (prop for prop in self.mean_curvature_torsion)

                streamlines = LazyStreamlines(points=gen_points, scalars=gen_scalars, properties=gen_properties)
                #streamlines.hdr[Field.NB_STREAMLINES] = len(self.points)

                streamline_utils.save(streamlines, f.name)
                streamlines_loaded = streamline_utils.load(f, None, lazy_load=False)

                hdr = streamlines_loaded.header
                assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))
                assert_equal(len(streamlines_loaded), len(self.points))

                points = streamlines_loaded.points
                assert_arrays_equal(points, self.points)

                # Check if we can iterate through the streamlines.
                for point, scalar, prop in streamlines_loaded:
                    pass

    def test_save_file_from_function(self):
        for ext in streamline_utils.FORMATS.keys():
            # With scalars
            with tempfile.NamedTemporaryFile(mode="w+b", suffix=ext) as f:
                gen_points = lambda: (point for point in self.points)
                gen_scalars = lambda: (scalar for scalar in self.colors)
                gen_properties = lambda: (prop for prop in self.mean_curvature_torsion)

                streamlines = LazyStreamlines(points=gen_points, scalars=gen_scalars, properties=gen_properties)
                #streamlines.hdr[Field.NB_STREAMLINES] = len(self.points)

                streamline_utils.save(streamlines, f.name)
                streamlines_loaded = streamline_utils.load(f, None, lazy_load=False)

                hdr = streamlines_loaded.header
                assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))
                assert_equal(len(streamlines_loaded), len(self.points))

                points = streamlines_loaded.points
                assert_arrays_equal(points, self.points)

                # Check if we can iterate through the streamlines.
                for point, scalar, prop in streamlines_loaded:
                    pass
