import os
import unittest
import numpy as np

from nibabel.externals.six import BytesIO

from nibabel.testing import assert_arrays_equal
from nose.tools import assert_equal, assert_raises

import nibabel as nib
from nibabel.streamlines.base_format import Streamlines
from nibabel.streamlines.base_format import DataError, HeaderError
from nibabel.streamlines.header import Field
from nibabel.streamlines.trk import TrkFile

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


class TestTRK(unittest.TestCase):

    def setUp(self):
        self.empty_trk_filename = os.path.join(DATA_PATH, "empty.trk")
        self.simple_trk_filename = os.path.join(DATA_PATH, "simple.trk")
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

    def test_load_empty_file(self):
        empty_trk = nib.streamlines.load(self.empty_trk_filename, lazy_load=False)

        hdr = empty_trk.get_header()
        assert_equal(hdr[Field.NB_STREAMLINES], 0)
        assert_equal(len(empty_trk), 0)

        points = empty_trk.points
        assert_equal(len(points), 0)

        scalars = empty_trk.scalars
        assert_equal(len(scalars), 0)

        properties = empty_trk.properties
        assert_equal(len(properties), 0)

        # Check if we can iterate through the streamlines.
        for point, scalar, prop in empty_trk:
            pass

    def test_load_simple_file(self):
        simple_trk = nib.streamlines.load(self.simple_trk_filename, lazy_load=False)

        hdr = simple_trk.get_header()
        assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))
        assert_equal(len(simple_trk), len(self.points))

        points = simple_trk.points
        assert_arrays_equal(points, self.points)

        scalars = simple_trk.scalars
        assert_equal(len(scalars), len(self.points))

        properties = simple_trk.properties
        assert_equal(len(properties), len(self.points))

        # Check if we can iterate through the streamlines.
        for point, scalar, prop in simple_trk:
            pass

        # Test lazy_load
        simple_trk = nib.streamlines.load(self.simple_trk_filename, lazy_load=True)

        hdr = simple_trk.get_header()
        assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))
        assert_equal(len(simple_trk), len(self.points))

        points = simple_trk.points
        assert_arrays_equal(points, self.points)

        scalars = simple_trk.scalars
        assert_equal(len(list(scalars)), len(self.points))

        properties = simple_trk.properties
        assert_equal(len(list(properties)), len(self.points))

        # Check if we can iterate through the streamlines.
        for point, scalar, prop in simple_trk:
            pass

    def test_load_complex_file(self):
        complex_trk = nib.streamlines.load(self.complex_trk_filename, lazy_load=False)

        hdr = complex_trk.get_header()
        assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))
        assert_equal(len(complex_trk), len(self.points))

        points = complex_trk.points
        assert_arrays_equal(points, self.points)

        scalars = complex_trk.scalars
        assert_arrays_equal(scalars, self.colors)

        properties = complex_trk.properties
        assert_arrays_equal(properties, self.mean_curvature_torsion)

        # Check if we can iterate through the streamlines.
        for point, scalar, prop in complex_trk:
            pass

        complex_trk = nib.streamlines.load(self.complex_trk_filename, lazy_load=True)

        hdr = complex_trk.get_header()
        assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))
        assert_equal(len(complex_trk), len(self.points))

        points = complex_trk.points
        assert_arrays_equal(points, self.points)

        scalars = complex_trk.scalars
        assert_arrays_equal(scalars, self.colors)

        properties = complex_trk.properties
        assert_arrays_equal(properties, self.mean_curvature_torsion)

        # Check if we can iterate through the streamlines.
        for point, scalar, prop in complex_trk:
            pass

    def test_write_simple_file(self):
        streamlines = Streamlines(self.points)

        simple_trk_file = BytesIO()
        TrkFile.save(streamlines, simple_trk_file)

        simple_trk_file.seek(0, os.SEEK_SET)

        simple_trk = nib.streamlines.load(simple_trk_file, lazy_load=False)

        hdr = simple_trk.get_header()
        assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))
        assert_equal(len(simple_trk), len(self.points))

        points = simple_trk.points
        assert_arrays_equal(points, self.points)

        scalars = simple_trk.scalars
        assert_equal(len(scalars), len(self.points))

        properties = simple_trk.properties
        assert_equal(len(properties), len(self.points))

        # Check if we can iterate through the streamlines.
        for point, scalar, prop in simple_trk:
            pass

    def test_write_complex_file(self):
        # With scalars
        streamlines = Streamlines(self.points, scalars=self.colors)

        trk_file = BytesIO()
        TrkFile.save(streamlines, trk_file)

        trk_file.seek(0, os.SEEK_SET)

        trk = nib.streamlines.load(trk_file, lazy_load=False)

        hdr = trk.get_header()
        assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))
        assert_equal(len(trk), len(self.points))

        points = trk.points
        assert_arrays_equal(points, self.points)

        scalars = trk.scalars
        assert_arrays_equal(scalars, self.colors)

        properties = trk.properties
        assert_equal(len(properties), len(self.points))

        # Check if we can iterate through the streamlines.
        for point, scalar, prop in trk:
            pass

        # With properties
        streamlines = Streamlines(self.points, properties=self.mean_curvature_torsion)

        trk_file = BytesIO()
        TrkFile.save(streamlines, trk_file)

        trk_file.seek(0, os.SEEK_SET)

        trk = nib.streamlines.load(trk_file, lazy_load=False)

        hdr = trk.get_header()
        assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))
        assert_equal(len(trk), len(self.points))

        points = trk.points
        assert_arrays_equal(points, self.points)

        scalars = trk.scalars
        assert_equal(len(scalars), len(self.points))

        properties = trk.properties
        assert_arrays_equal(properties, self.mean_curvature_torsion)

        # Check if we can iterate through the streamlines.
        for point, scalar, prop in trk:
            pass

        # With scalars and properties
        streamlines = Streamlines(self.points, scalars=self.colors, properties=self.mean_curvature_torsion)

        trk_file = BytesIO()
        TrkFile.save(streamlines, trk_file)

        trk_file.seek(0, os.SEEK_SET)

        trk = nib.streamlines.load(trk_file, lazy_load=False)

        hdr = trk.get_header()
        assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))
        assert_equal(len(trk), len(self.points))

        points = trk.points
        assert_arrays_equal(points, self.points)

        scalars = trk.scalars
        assert_arrays_equal(scalars, self.colors)

        properties = trk.properties
        assert_arrays_equal(properties, self.mean_curvature_torsion)

        # Check if we can iterate through the streamlines.
        for point, scalar, prop in trk:
            pass

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

    def test_write_file_from_generator(self):
        gen_points = (point for point in self.points)
        gen_scalars = (scalar for scalar in self.colors)
        gen_properties = (prop for prop in self.mean_curvature_torsion)

        assert_raises(HeaderError, Streamlines, points=gen_points, scalars=gen_scalars, properties=gen_properties)

        hdr = {Field.NB_STREAMLINES: len(self.points)}
        streamlines = Streamlines(points=gen_points, scalars=gen_scalars, properties=gen_properties, hdr=hdr)

        trk_file = BytesIO()
        TrkFile.save(streamlines, trk_file)

        trk_file.seek(0, os.SEEK_SET)

        trk = nib.streamlines.load(trk_file, lazy_load=False)

        hdr = trk.get_header()
        assert_equal(hdr[Field.NB_STREAMLINES], len(self.points))
        assert_equal(len(trk), len(self.points))

        points = trk.points
        assert_arrays_equal(points, self.points)

        scalars = trk.scalars
        assert_arrays_equal(scalars, self.colors)

        properties = trk.properties
        assert_arrays_equal(properties, self.mean_curvature_torsion)

        # Check if we can iterate through the streamlines.
        for point, scalar, prop in trk:
            pass
