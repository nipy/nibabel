import os
import unittest
import numpy as np

from nibabel.testing import assert_arrays_equal
from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_array_equal

from nibabel.streamlines.base_format import Streamlines, LazyStreamlines
from nibabel.streamlines.base_format import HeaderError
from nibabel.streamlines.header import Field

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


class TestLazyStreamlines(unittest.TestCase):

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
        streamlines = LazyStreamlines()

        # LazyStreamlines have a default header when created from arrays:
        #  NB_STREAMLINES, NB_SCALARS_PER_POINT, NB_PROPERTIES_PER_STREAMLINE
        #  and VOXEL_TO_WORLD.
        hdr = streamlines.header
        assert_equal(len(hdr), 1)
        assert_equal(len(streamlines), 0)

        # Check if we can iterate through the streamlines.
        for point, scalar, prop in streamlines:
            pass

        # Only points
        streamlines = LazyStreamlines(points=self.points)
        assert_equal(len(streamlines), len(self.points))
        assert_arrays_equal(streamlines.points, self.points)

        # Check if we can iterate through the streamlines.
        for point, scalar, prop in streamlines:
            pass

        # Only scalars
        streamlines = LazyStreamlines(scalars=self.colors)
        assert_arrays_equal(streamlines.scalars, self.colors)

        # TODO: is it a faulty behavior?
        assert_equal(len(list(streamlines)), len(self.colors))

        # Points, scalars and properties
        streamlines = LazyStreamlines(self.points, self.colors, self.mean_curvature_torsion)
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

        streamlines = LazyStreamlines(points, scalars, properties)

        # LazyStreamlines object does not support indexing.
        assert_raises(AttributeError, streamlines.__getitem__, 0)

        assert_arrays_equal(streamlines.points, self.points)
        assert_arrays_equal(streamlines.scalars, self.colors)
        assert_arrays_equal(streamlines.properties, self.mean_curvature_torsion)

        # Have been consumed
        assert_equal(len(list(streamlines)), 0)
        assert_equal(len(list(streamlines.points)), 0)
        assert_equal(len(list(streamlines.scalars)), 0)
        assert_equal(len(list(streamlines.properties)), 0)

        # Test function len
        points = (x for x in self.points)
        streamlines = LazyStreamlines(points, scalars, properties)

        # This will consume generator `points`.
        # Note this will produce a warning message.
        assert_equal(len(streamlines), len(self.points))
        assert_equal(len(streamlines), 0)

        # It will use `Field.NB_STREAMLINES` if it is in the streamlines header
        # Note this won't produce a warning message.
        streamlines.header[Field.NB_STREAMLINES] = len(self.points)
        assert_equal(len(streamlines), len(self.points))

    def test_streamlines_creation_from_functions(self):
        # Points, scalars and properties
        points = lambda: (x for x in self.points)
        scalars = lambda: (x for x in self.colors)
        properties = lambda: (x for x in self.mean_curvature_torsion)

        streamlines = LazyStreamlines(points, scalars, properties)

        # LazyStreamlines object does not support indexing.
        assert_raises(AttributeError, streamlines.__getitem__, 0)

        assert_arrays_equal(streamlines.points, self.points)
        assert_arrays_equal(streamlines.scalars, self.colors)
        assert_arrays_equal(streamlines.properties, self.mean_curvature_torsion)

        # Have been consumed but lambda functions get re-called.
        assert_equal(len(list(streamlines)), len(self.points))
        assert_arrays_equal(streamlines.points, self.points)
        assert_arrays_equal(streamlines.scalars, self.colors)
        assert_arrays_equal(streamlines.properties, self.mean_curvature_torsion)

        # Test function `len`
        # Calling `len` will create a new generator each time.
        # Note this will produce a warning message.
        assert_equal(len(streamlines), len(self.points))
        assert_equal(len(streamlines), len(self.points))

        # It will use `Field.NB_STREAMLINES` if it is in the streamlines header
        # Note this won't produce a warning message.
        streamlines.header[Field.NB_STREAMLINES] = len(self.points)
        assert_equal(len(streamlines), len(self.points))

    def test_len(self):
        # Points, scalars and properties
        points = lambda: (x for x in self.points)

        # Function `len` is computed differently depending on available information.
        # When `points` is a list, `len` will use `len(points)`.
        streamlines = LazyStreamlines(points=self.points)
        assert_equal(len(streamlines), len(self.points))

        # When `points` is a generator, `len` will iterate through the streamlines
        # and consume the generator.
        # TODO: check that it has raised a warning message.
        streamlines = LazyStreamlines(points=points())
        assert_equal(len(streamlines), len(self.points))
        assert_equal(len(streamlines), 0)

        # When `points` is a callable object that creates a generator, `len` will iterate
        # through the streamlines.
        # TODO: check that it has raised a warning message.
        streamlines = LazyStreamlines(points=points)
        assert_equal(len(streamlines), len(self.points))
        assert_equal(len(streamlines), len(self.points))


        # No matter what `points` is, if `Field.NB_STREAMLINES` is set in the header
        # `len` returns that value. If not and `count` argument is specified, `len`
        # will use that information to return a value.
        # TODO: check that no warning messages are raised.
        for pts in [self.points, points(), points]:
            # `Field.NB_STREAMLINES` is set in the header.
            streamlines = LazyStreamlines(points=pts)
            streamlines.header[Field.NB_STREAMLINES] = 42
            assert_equal(len(streamlines), 42)

            # `count` is an integer.
            streamlines = LazyStreamlines(points=pts, count=42)
            assert_equal(len(streamlines), 42)

            # `count` argument is a callable object.
            nb_calls = [0]
            def count():
                nb_calls[0] += 1
                return 42

            streamlines = LazyStreamlines(points=points, count=count)
            assert_equal(len(streamlines), 42)
            assert_equal(len(streamlines), 42)
            # Check that the callable object is only called once (caching).
            assert_equal(nb_calls[0], 1)


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

        # Streamlines have a default header:
        #  NB_STREAMLINES, NB_SCALARS_PER_POINT, NB_PROPERTIES_PER_STREAMLINE
        #  and VOXEL_TO_WORLD.
        hdr = streamlines.header
        assert_equal(len(hdr), 4)

        # Check if we can iterate through the streamlines.
        for points, scalars, props in streamlines:
            pass

        # Only points
        streamlines = Streamlines(points=self.points)
        assert_equal(len(streamlines), len(self.points))
        assert_arrays_equal(streamlines.points, self.points)

        # Check if we can iterate through the streamlines.
        for points, scalars, props in streamlines:
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

        # Retrieves streamlines by their index
        for i, (points, scalars, props) in enumerate(streamlines):
            points_i, scalars_i, props_i = streamlines[i]
            assert_array_equal(points_i, points)
            assert_array_equal(scalars_i, scalars)
            assert_array_equal(props_i, props)

    def test_streamlines_creation_from_generators(self):
        # Points, scalars and properties
        points = (x for x in self.points)
        scalars = (x for x in self.colors)
        properties = (x for x in self.mean_curvature_torsion)

        # To create streamlines from generators use LazyStreamlines.
        assert_raises(TypeError, Streamlines, points, scalars, properties)

    def test_streamlines_creation_from_functions(self):
        # Points, scalars and properties
        points = lambda: (x for x in self.points)
        scalars = lambda: (x for x in self.colors)
        properties = lambda: (x for x in self.mean_curvature_torsion)

        # To create streamlines from functions use LazyStreamlines.
        assert_raises(TypeError, Streamlines, points, scalars, properties)
