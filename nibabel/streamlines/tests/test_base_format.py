import os
import unittest
import numpy as np
import warnings

from nibabel.testing import assert_arrays_equal
from nibabel.testing import suppress_warnings, clear_and_catch_warnings
from nose.tools import assert_equal, assert_raises, assert_true
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nibabel.externals.six.moves import zip

from .. import base_format
from ..base_format import Streamlines, LazyStreamlines
from ..base_format import UsageWarning

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

        self.nb_streamlines = len(self.points)
        self.nb_scalars_per_point = self.colors[0].shape[1]
        self.nb_properties_per_streamline = len(self.mean_curvature_torsion[0])

    def test_streamlines_creation_from_arrays(self):
        # Empty
        streamlines = Streamlines()
        assert_equal(len(streamlines), 0)
        assert_arrays_equal(streamlines.points, [])
        assert_arrays_equal(streamlines.scalars, [])
        assert_arrays_equal(streamlines.properties, [])

        # Check if we can iterate through the streamlines.
        for streamline in streamlines:
            pass

        # Only points
        streamlines = Streamlines(points=self.points)
        assert_equal(len(streamlines), len(self.points))
        assert_arrays_equal(streamlines.points, self.points)
        assert_arrays_equal(streamlines.scalars, [])
        assert_arrays_equal(streamlines.properties, [])

        # Check if we can iterate through the streamlines.
        for streamline in streamlines:
            pass

        # Points, scalars and properties
        streamlines = Streamlines(self.points, self.colors, self.mean_curvature_torsion)
        assert_equal(len(streamlines), len(self.points))
        assert_arrays_equal(streamlines.points, self.points)
        assert_arrays_equal(streamlines.scalars, self.colors)
        assert_arrays_equal(streamlines.properties, self.mean_curvature_torsion)

        # Check if we can iterate through the streamlines.
        for streamline in streamlines:
            pass

        #streamlines = Streamlines(self.points, scalars)

    def test_streamlines_getter(self):
        # Streamlines with only points
        streamlines = Streamlines(points=self.points)

        selected_streamlines = streamlines[::2]
        assert_equal(len(selected_streamlines), (len(self.points)+1)//2)

        assert_arrays_equal(selected_streamlines.points, self.points[::2])
        assert_equal(sum(map(len, selected_streamlines.scalars)), 0)
        assert_equal(sum(map(len, selected_streamlines.properties)), 0)

        # Streamlines with points, scalars and properties
        streamlines = Streamlines(self.points, self.colors, self.mean_curvature_torsion)

        # Retrieve streamlines by their index
        for i, streamline in enumerate(streamlines):
            assert_array_equal(streamline.points, streamlines[i].points)
            assert_array_equal(streamline.scalars, streamlines[i].scalars)
            assert_array_equal(streamline.properties, streamlines[i].properties)

        # Use slicing
        r_streamlines = streamlines[::-1]
        assert_arrays_equal(r_streamlines.points, self.points[::-1])
        assert_arrays_equal(r_streamlines.scalars, self.colors[::-1])
        assert_arrays_equal(r_streamlines.properties, self.mean_curvature_torsion[::-1])

    def test_streamlines_creation_from_coroutines(self):
        # Points, scalars and properties
        points = lambda: (x for x in self.points)
        scalars = lambda: (x for x in self.colors)
        properties = lambda: (x for x in self.mean_curvature_torsion)

        # To create streamlines from coroutines use `LazyStreamlines`.
        assert_raises(TypeError, Streamlines, points, scalars, properties)

    def test_to_world_space(self):
        streamlines = Streamlines(self.points)

        # World space is (RAS+) with voxel size of 2x3x4mm.
        streamlines.header.voxel_sizes = (2, 3, 4)

        new_streamlines = streamlines.to_world_space()
        for new_pts, pts in zip(new_streamlines.points, self.points):
            for dim, size in enumerate(streamlines.header.voxel_sizes):
                assert_array_almost_equal(new_pts[:, dim], size*pts[:, dim])

    def test_header(self):
        # Empty Streamlines, with default header
        streamlines = Streamlines()
        assert_equal(streamlines.header.nb_streamlines, 0)
        assert_equal(streamlines.header.nb_scalars_per_point, 0)
        assert_equal(streamlines.header.nb_properties_per_streamline, 0)
        assert_array_equal(streamlines.header.voxel_sizes, (1, 1, 1))
        assert_array_equal(streamlines.header.to_world_space, np.eye(4))
        assert_equal(streamlines.header.extra, {})

        streamlines = Streamlines(self.points, self.colors, self.mean_curvature_torsion)

        assert_equal(streamlines.header.nb_streamlines, len(self.points))
        assert_equal(streamlines.header.nb_scalars_per_point, self.colors[0].shape[1])
        assert_equal(streamlines.header.nb_properties_per_streamline, self.mean_curvature_torsion[0].shape[0])

        # Modifying voxel_sizes should be reflected in to_world_space
        streamlines.header.voxel_sizes = (2, 3, 4)
        assert_array_equal(streamlines.header.voxel_sizes, (2, 3, 4))
        assert_array_equal(np.diag(streamlines.header.to_world_space), (2, 3, 4, 1))

        # Modifying scaling of to_world_space should be reflected in voxel_sizes
        streamlines.header.to_world_space = np.diag([4, 3, 2, 1])
        assert_array_equal(streamlines.header.voxel_sizes, (4, 3, 2))
        assert_array_equal(streamlines.header.to_world_space, np.diag([4, 3, 2, 1]))

        # Test that we can run __repr__ without error.
        repr(streamlines.header)


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

        self.nb_streamlines = len(self.points)
        self.nb_scalars_per_point = self.colors[0].shape[1]
        self.nb_properties_per_streamline = len(self.mean_curvature_torsion[0])

    def test_lazy_streamlines_creation(self):
        # To create streamlines from arrays use `Streamlines`.
        assert_raises(TypeError, LazyStreamlines, self.points)

        # Points, scalars and properties
        points = (x for x in self.points)
        scalars = (x for x in self.colors)
        properties = (x for x in self.mean_curvature_torsion)

        # Creating LazyStreamlines from generators is not allowed as
        # generators get exhausted and are not reusable unline coroutines.
        assert_raises(TypeError, LazyStreamlines, points)
        assert_raises(TypeError, LazyStreamlines, self.points, scalars)
        assert_raises(TypeError, LazyStreamlines, properties_func=properties)

        # Empty `LazyStreamlines`
        streamlines = LazyStreamlines()
        with suppress_warnings():
            assert_equal(len(streamlines), 0)
        assert_arrays_equal(streamlines.points, [])
        assert_arrays_equal(streamlines.scalars, [])
        assert_arrays_equal(streamlines.properties, [])

        # Check if we can iterate through the streamlines.
        for streamline in streamlines:
            pass

        # Points, scalars and properties
        points = lambda: (x for x in self.points)
        scalars = lambda: (x for x in self.colors)
        properties = lambda: (x for x in self.mean_curvature_torsion)

        streamlines = LazyStreamlines(points, scalars, properties)
        with suppress_warnings():
            assert_equal(len(streamlines), self.nb_streamlines)

        # Coroutines get re-called and creates new iterators.
        assert_arrays_equal(streamlines.points, self.points)
        assert_arrays_equal(streamlines.scalars, self.colors)
        assert_arrays_equal(streamlines.properties, self.mean_curvature_torsion)
        assert_arrays_equal(streamlines.points, self.points)
        assert_arrays_equal(streamlines.scalars, self.colors)
        assert_arrays_equal(streamlines.properties, self.mean_curvature_torsion)

        # Create `LazyStreamlines` from a coroutine yielding 3-tuples
        data = lambda: (x for x in zip(self.points, self.colors, self.mean_curvature_torsion))

        streamlines = LazyStreamlines.create_from_data(data)
        with suppress_warnings():
            assert_equal(len(streamlines), self.nb_streamlines)
        assert_arrays_equal(streamlines.points, self.points)
        assert_arrays_equal(streamlines.scalars, self.colors)
        assert_arrays_equal(streamlines.properties, self.mean_curvature_torsion)

        # Check if we can iterate through the streamlines.
        for streamline in streamlines:
            pass

    def test_lazy_streamlines_indexing(self):
        points = lambda: (x for x in self.points)
        scalars = lambda: (x for x in self.colors)
        properties = lambda: (x for x in self.mean_curvature_torsion)

        # By default, `LazyStreamlines` object does not support indexing.
        streamlines = LazyStreamlines(points, scalars, properties)
        assert_raises(AttributeError, streamlines.__getitem__, 0)

        # Create a `LazyStreamlines` object with indexing support.
        def getitem_without_properties(idx):
            if isinstance(idx, int) or isinstance(idx, np.integer):
                return self.points[idx], self.colors[idx]

            return list(zip(self.points[idx], self.colors[idx]))

        streamlines = LazyStreamlines(points, scalars, properties, getitem_without_properties)
        points, scalars = streamlines[0]
        assert_array_equal(points, self.points[0])
        assert_array_equal(scalars, self.colors[0])

        points, scalars = zip(*streamlines[::-1])
        assert_arrays_equal(points, self.points[::-1])
        assert_arrays_equal(scalars, self.colors[::-1])

        points, scalars = zip(*streamlines[:-1])
        assert_arrays_equal(points, self.points[:-1])
        assert_arrays_equal(scalars, self.colors[:-1])

    def test_lazy_streamlines_len(self):
        points = lambda: (x for x in self.points)
        scalars = lambda: (x for x in self.colors)
        properties = lambda: (x for x in self.mean_curvature_torsion)

        with clear_and_catch_warnings(record=True, modules=[base_format]) as w:
            warnings.simplefilter("always")  # Always trigger warnings.

            # Calling `len` will create new generators each time.
            streamlines = LazyStreamlines(points, scalars, properties)
            # This should produce a warning message.
            assert_equal(len(streamlines), self.nb_streamlines)
            assert_equal(len(w), 1)

            streamlines = LazyStreamlines(points, scalars, properties)
            # This should still produce a warning message.
            assert_equal(len(streamlines), self.nb_streamlines)
            assert_equal(len(w), 2)
            assert_true(issubclass(w[-1].category, UsageWarning))

            # This should *not* produce a warning.
            assert_equal(len(streamlines), self.nb_streamlines)
            assert_equal(len(w), 2)

        with clear_and_catch_warnings(record=True, modules=[base_format]) as w:
            # Once we iterated through the streamlines, we know the length.
            streamlines = LazyStreamlines(points, scalars, properties)
            assert_true(streamlines.header.nb_streamlines is None)
            for streamline in streamlines:
                pass

            assert_equal(streamlines.header.nb_streamlines, len(self.points))
            # This should *not* produce a warning.
            assert_equal(len(streamlines), len(self.points))
            assert_equal(len(w), 0)

        with clear_and_catch_warnings(record=True, modules=[base_format]) as w:
            # It first checks if number of streamlines is in the header.
            streamlines = LazyStreamlines(points, scalars, properties)
            streamlines.header.nb_streamlines = 1234
            # This should *not* produce a warning.
            assert_equal(len(streamlines), 1234)
            assert_equal(len(w), 0)

    def test_lazy_streamlines_header(self):
        # Empty `LazyStreamlines`, with default header
        streamlines = LazyStreamlines()
        assert_true(streamlines.header.nb_streamlines is None)
        assert_equal(streamlines.header.nb_scalars_per_point, 0)
        assert_equal(streamlines.header.nb_properties_per_streamline, 0)
        assert_array_equal(streamlines.header.voxel_sizes, (1, 1, 1))
        assert_array_equal(streamlines.header.to_world_space, np.eye(4))
        assert_equal(streamlines.header.extra, {})

        points = lambda: (x for x in self.points)
        scalars = lambda: (x for x in self.colors)
        properties = lambda: (x for x in self.mean_curvature_torsion)
        streamlines = LazyStreamlines(points)
        header = streamlines.header

        assert_equal(header.nb_scalars_per_point, 0)
        streamlines.scalars = scalars
        assert_equal(header.nb_scalars_per_point, self.nb_scalars_per_point)

        assert_equal(header.nb_properties_per_streamline, 0)
        streamlines.properties = properties
        assert_equal(header.nb_properties_per_streamline, self.nb_properties_per_streamline)
