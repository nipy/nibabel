import os
import unittest
import numpy as np

from nibabel.testing import assert_arrays_equal
from nose.tools import assert_equal, assert_raises

from nibabel.streamlines.base_format import Streamlines
from nibabel.streamlines.base_format import HeaderError
from nibabel.streamlines.header import Field

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


class TestStreamlines(unittest.TestCase):

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

    def test_streamlines_creation_from_arrays(self):
        # Empty
        streamlines = Streamlines()
        assert_equal(len(streamlines), 0)

        # TODO: Should Streamlines have a default header? It could have
        #  NB_STREAMLINES, NB_SCALARS_PER_POINT and NB_PROPERTIES_PER_STREAMLINE
        #  already set.
        hdr = streamlines.get_header()
        assert_equal(len(hdr), 0)

        # Check if we can iterate through the streamlines.
        for point, scalar, prop in streamlines:
            pass

        # Only points
        streamlines = Streamlines(points=self.points)
        assert_equal(len(streamlines), len(self.points))
        assert_arrays_equal(streamlines.points, self.points)

        # Check if we can iterate through the streamlines.
        for point, scalar, prop in streamlines:
            pass

        # Only scalars
        streamlines = Streamlines(scalars=self.colors)
        assert_equal(len(streamlines), 0)
        assert_arrays_equal(streamlines.scalars, self.colors)

        # TODO: is it a faulty behavior?
        assert_equal(len(list(streamlines)), len(self.colors))

        # Points, scalars and properties
        streamlines = Streamlines(self.points, self.colors, self.mean_curvature_torsion)
        assert_equal(len(streamlines), len(self.points))
        assert_arrays_equal(streamlines.points, self.points)
        assert_arrays_equal(streamlines.scalars, self.colors)
        assert_arrays_equal(streamlines.properties, self.mean_curvature_torsion)

        # Check if we can iterate through the streamlines.
        for point, scalar, prop in streamlines:
            pass

    def test_streamlines_creation_from_generators(self):
        # Points, scalars and properties
        points = (x for x in self.points)
        scalars = (x for x in self.colors)
        properties = (x for x in self.mean_curvature_torsion)

        assert_raises(HeaderError, Streamlines, points, scalars, properties)

        hdr = {Field.NB_STREAMLINES: len(self.points)}
        streamlines = Streamlines(points, scalars, properties, hdr)
        assert_equal(len(streamlines), len(self.points))
        assert_arrays_equal(streamlines.points, self.points)
        assert_arrays_equal(streamlines.scalars, self.colors)
        assert_arrays_equal(streamlines.properties, self.mean_curvature_torsion)

        # Have been consumed
        assert_equal(len(list(streamlines)), 0)
        assert_equal(len(list(streamlines.points)), 0)
        assert_equal(len(list(streamlines.scalars)), 0)
        assert_equal(len(list(streamlines.properties)), 0)

    def test_streamlines_creation_from_functions(self):
        # Points, scalars and properties
        points = lambda: (x for x in self.points)
        scalars = lambda: (x for x in self.colors)
        properties = lambda: (x for x in self.mean_curvature_torsion)

        assert_raises(HeaderError, Streamlines, points, scalars, properties)

        hdr = {Field.NB_STREAMLINES: len(self.points)}
        streamlines = Streamlines(points, scalars, properties, hdr)
        assert_equal(len(streamlines), len(self.points))
        assert_arrays_equal(streamlines.points, self.points)
        assert_arrays_equal(streamlines.scalars, self.colors)
        assert_arrays_equal(streamlines.properties, self.mean_curvature_torsion)

        # Have been consumed but lambda functions get re-called.
        assert_equal(len(list(streamlines)), len(self.points))
        assert_arrays_equal(streamlines.points, self.points)
        assert_arrays_equal(streamlines.scalars, self.colors)
        assert_arrays_equal(streamlines.properties, self.mean_curvature_torsion)
