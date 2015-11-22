import unittest
import numpy as np
import warnings

from nibabel.testing import assert_arrays_equal, isiterable
from nibabel.testing import suppress_warnings, clear_and_catch_warnings
from nose.tools import assert_equal, assert_raises, assert_true
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nibabel.externals.six.moves import zip

from .. import tractogram as module_tractogram
from ..tractogram import UsageWarning
from ..tractogram import TractogramItem, Tractogram, LazyTractogram


class TestTractogramItem(unittest.TestCase):

    def test_creating_tractogram_item(self):
        rng = np.random.RandomState(42)
        streamline = rng.rand(rng.randint(10, 50), 3)
        colors = rng.rand(len(streamline), 3)
        mean_curvature = 1.11
        mean_color = np.array([0, 1, 0], dtype="f4")

        data_for_streamline = {"mean_curvature": mean_curvature,
                               "mean_color": mean_color}

        data_for_points = {"colors": colors}

        # Create a tractogram item with a streamline, data.
        t = TractogramItem(streamline, data_for_streamline, data_for_points)
        assert_equal(len(t), len(streamline))
        assert_array_equal(t.streamline, streamline)
        assert_array_equal(list(t), streamline)
        assert_array_equal(t.data_for_streamline['mean_curvature'],
                           mean_curvature)
        assert_array_equal(t.data_for_streamline['mean_color'],
                           mean_color)
        assert_array_equal(t.data_for_points['colors'],
                           colors)


class TestTractogram(unittest.TestCase):

    def setUp(self):
        self.streamlines = [np.arange(1*3, dtype="f4").reshape((1, 3)),
                            np.arange(2*3, dtype="f4").reshape((2, 3)),
                            np.arange(5*3, dtype="f4").reshape((5, 3))]

        self.colors = [np.array([(1, 0, 0)]*1, dtype="f4"),
                       np.array([(0, 1, 0)]*2, dtype="f4"),
                       np.array([(0, 0, 1)]*5, dtype="f4")]

        self.mean_curvature = np.array([1.11, 2.11, 3.11], dtype="f4")
        self.mean_color = np.array([[0, 1, 0],
                                    [0, 0, 1],
                                    [1, 0, 0]], dtype="f4")

        self.nb_streamlines = len(self.streamlines)

    def test_tractogram_creation(self):
        # Create an empty tractogram.
        tractogram = Tractogram()
        assert_equal(len(tractogram), 0)
        assert_arrays_equal(tractogram.streamlines, [])
        assert_equal(tractogram.data_per_streamline, {})
        assert_equal(tractogram.data_per_point, {})
        assert_true(isiterable(tractogram))

        # Create a tractogram with only streamlines
        tractogram = Tractogram(streamlines=self.streamlines)
        assert_equal(len(tractogram), len(self.streamlines))
        assert_arrays_equal(tractogram.streamlines, self.streamlines)
        assert_equal(tractogram.data_per_streamline, {})
        assert_equal(tractogram.data_per_point, {})
        assert_true(isiterable(tractogram))

        # Create a tractogram with streamlines and other data.
        tractogram = Tractogram(
            self.streamlines,
            data_per_streamline={'mean_curvature': self.mean_curvature,
                                 'mean_color': self.mean_color},
            data_per_point={'colors': self.colors})

        assert_equal(len(tractogram), len(self.streamlines))
        assert_arrays_equal(tractogram.streamlines, self.streamlines)
        assert_arrays_equal(tractogram.data_per_streamline['mean_curvature'],
                            self.mean_curvature)
        assert_arrays_equal(tractogram.data_per_streamline['mean_color'],
                            self.mean_color)
        assert_arrays_equal(tractogram.data_per_point['colors'],
                            self.colors)

        assert_true(isiterable(tractogram))

        # Inconsistent number of scalars between streamlines
        wrong_data = [[(1, 0, 0)]*1,
                      [(0, 1, 0), (0, 1)],
                      [(0, 0, 1)]*5]

        data_per_point = {'wrong_data': wrong_data}
        assert_raises(ValueError, Tractogram, self.streamlines,
                      data_per_point=data_per_point)

        # Inconsistent number of scalars between streamlines
        wrong_data = [[(1, 0, 0)]*1,
                      [(0, 1)]*2,
                      [(0, 0, 1)]*5]

        data_per_point = {'wrong_data': wrong_data}
        assert_raises(ValueError, Tractogram, self.streamlines,
                      data_per_point=data_per_point)

    def test_tractogram_getter(self):
        # Tractogram with only streamlines
        tractogram = Tractogram(streamlines=self.streamlines)

        selected_tractogram = tractogram[::2]
        assert_equal(len(selected_tractogram), (len(self.streamlines)+1)//2)

        assert_arrays_equal(selected_tractogram.streamlines,
                            self.streamlines[::2])
        assert_equal(tractogram.data_per_streamline, {})
        assert_equal(tractogram.data_per_point, {})

        # Create a tractogram with streamlines and other data.
        tractogram = Tractogram(
            self.streamlines,
            data_per_streamline={'mean_curvature': self.mean_curvature,
                                 'mean_color': self.mean_color},
            data_per_point={'colors': self.colors})

        # Retrieve tractogram by their index
        for i, t in enumerate(tractogram):
            assert_array_equal(t.streamline, tractogram[i].streamline)
            assert_array_equal(t.data_for_points['colors'],
                               tractogram[i].data_for_points['colors'])

            assert_array_equal(t.data_for_streamline['mean_curvature'],
                               tractogram[i].data_for_streamline['mean_curvature'])

            assert_array_equal(t.data_for_streamline['mean_color'],
                               tractogram[i].data_for_streamline['mean_color'])

        # Use slicing
        r_tractogram = tractogram[::-1]
        assert_arrays_equal(r_tractogram.streamlines, self.streamlines[::-1])

        assert_arrays_equal(r_tractogram.data_per_streamline['mean_curvature'],
                            self.mean_curvature[::-1])
        assert_arrays_equal(r_tractogram.data_per_streamline['mean_color'],
                            self.mean_color[::-1])
        assert_arrays_equal(r_tractogram.data_per_point['colors'],
                            self.colors[::-1])

    def test_tractogram_add_new_data(self):
        # Tractogram with only streamlines
        tractogram = Tractogram(streamlines=self.streamlines)

        tractogram.data_per_streamline['mean_curvature'] = self.mean_curvature
        tractogram.data_per_streamline['mean_color'] = self.mean_color
        tractogram.data_per_point['colors'] = self.colors

        # Retrieve tractogram by their index
        for i, t in enumerate(tractogram):
            assert_array_equal(t.streamline, tractogram[i].streamline)
            assert_array_equal(t.data_for_points['colors'],
                               tractogram[i].data_for_points['colors'])

            assert_array_equal(t.data_for_streamline['mean_curvature'],
                               tractogram[i].data_for_streamline['mean_curvature'])

            assert_array_equal(t.data_for_streamline['mean_color'],
                               tractogram[i].data_for_streamline['mean_color'])

        # Use slicing
        r_tractogram = tractogram[::-1]
        assert_arrays_equal(r_tractogram.streamlines, self.streamlines[::-1])

        assert_arrays_equal(r_tractogram.data_per_streamline['mean_curvature'],
                            self.mean_curvature[::-1])
        assert_arrays_equal(r_tractogram.data_per_streamline['mean_color'],
                            self.mean_color[::-1])
        assert_arrays_equal(r_tractogram.data_per_point['colors'],
                            self.colors[::-1])


class TestLazyTractogram(unittest.TestCase):

    def setUp(self):
        self.streamlines = [np.arange(1*3, dtype="f4").reshape((1, 3)),
                            np.arange(2*3, dtype="f4").reshape((2, 3)),
                            np.arange(5*3, dtype="f4").reshape((5, 3))]

        self.colors = [np.array([(1, 0, 0)]*1, dtype="f4"),
                       np.array([(0, 1, 0)]*2, dtype="f4"),
                       np.array([(0, 0, 1)]*5, dtype="f4")]

        self.mean_curvature = np.array([1.11, 2.11, 3.11], dtype="f4")
        self.mean_color = np.array([[0, 1, 0],
                                    [0, 0, 1],
                                    [1, 0, 0]], dtype="f4")

        self.nb_streamlines = len(self.streamlines)

        self.colors_func = lambda: (x for x in self.colors)
        self.mean_curvature_func = lambda: (x for x in self.mean_curvature)
        self.mean_color_func = lambda: (x for x in self.mean_color)

    def test_lazy_tractogram_creation(self):
        # To create tractogram from arrays use `Tractogram`.
        assert_raises(TypeError, LazyTractogram, self.streamlines)

        # Streamlines and other data as generators
        streamlines = (x for x in self.streamlines)
        data_per_point = {"colors": (x for x in self.colors)}
        data_per_streamline = {'mean_curv': (x for x in self.mean_curvature),
                               'mean_color': (x for x in self.mean_color)}

        # Creating LazyTractogram with generators is not allowed as
        # generators get exhausted and are not reusable unlike coroutines.
        assert_raises(TypeError, LazyTractogram, streamlines)
        assert_raises(TypeError, LazyTractogram,
                      data_per_streamline=data_per_streamline)
        assert_raises(TypeError, LazyTractogram, self.streamlines,
                      data_per_point=data_per_point)

        # Empty `LazyTractogram`
        tractogram = LazyTractogram()
        assert_true(isiterable(tractogram))
        assert_equal(len(tractogram), 0)
        assert_arrays_equal(tractogram.streamlines, [])
        assert_equal(tractogram.data_per_point, {})
        assert_equal(tractogram.data_per_streamline, {})

        # Create tractogram with streamlines and other data
        streamlines = lambda: (x for x in self.streamlines)
        data_per_point = {"colors": self.colors_func}
        data_per_streamline = {'mean_curv': self.mean_curvature_func,
                               'mean_color': self.mean_color_func}

        tractogram = LazyTractogram(streamlines,
                                    data_per_streamline=data_per_streamline,
                                    data_per_point=data_per_point)

        assert_true(isiterable(tractogram))
        assert_equal(len(tractogram), self.nb_streamlines)

        # Coroutines get re-called and creates new iterators.
        for i in range(2):
            assert_arrays_equal(tractogram.streamlines, self.streamlines)
            assert_arrays_equal(tractogram.data_per_streamline['mean_curv'],
                                self.mean_curvature)
            assert_arrays_equal(tractogram.data_per_streamline['mean_color'],
                                self.mean_color)
            assert_arrays_equal(tractogram.data_per_point['colors'],
                                self.colors)

    def test_lazy_tractogram_create_from(self):
        # Create `LazyTractogram` from a coroutine yielding nothing (i.e empty).
        _empty_data_gen = lambda: iter([])

        tractogram = LazyTractogram.create_from(_empty_data_gen)
        assert_true(isiterable(tractogram))
        assert_equal(len(tractogram), 0)
        assert_arrays_equal(tractogram.streamlines, [])
        assert_equal(tractogram.data_per_point, {})
        assert_equal(tractogram.data_per_streamline, {})

        # Create `LazyTractogram` from a coroutine yielding TractogramItem
        def _data_gen():
            for d in zip(self.streamlines, self.colors,
                         self.mean_curvature, self.mean_color):
                data_for_points = {'colors': d[1]}
                data_for_streamline = {'mean_curv': d[2],
                                       'mean_color': d[3]}
                yield TractogramItem(d[0], data_for_streamline, data_for_points)

        tractogram = LazyTractogram.create_from(_data_gen)
        assert_true(isiterable(tractogram))
        assert_equal(len(tractogram), self.nb_streamlines)
        assert_arrays_equal(tractogram.streamlines, self.streamlines)
        assert_arrays_equal(tractogram.data_per_streamline['mean_curv'],
                            self.mean_curvature)
        assert_arrays_equal(tractogram.data_per_streamline['mean_color'],
                            self.mean_color)
        assert_arrays_equal(tractogram.data_per_point['colors'],
                            self.colors)

    def test_lazy_tractogram_indexing(self):
        streamlines = lambda: (x for x in self.streamlines)
        data_per_point = {"colors": self.colors_func}
        data_per_streamline = {'mean_curv': self.mean_curvature_func,
                               'mean_color': self.mean_color_func}

        # By default, `LazyTractogram` object does not support indexing.
        tractogram = LazyTractogram(streamlines,
                                    data_per_streamline=data_per_streamline,
                                    data_per_point=data_per_point)
        assert_raises(AttributeError, tractogram.__getitem__, 0)

    def test_lazy_tractogram_len(self):
        streamlines = lambda: (x for x in self.streamlines)
        data_per_point = {"colors": self.colors_func}
        data_per_streamline = {'mean_curv': self.mean_curvature_func,
                               'mean_color': self.mean_color_func}

        modules = [module_tractogram]  # Modules for which to catch warnings.
        with clear_and_catch_warnings(record=True, modules=modules) as w:
            warnings.simplefilter("always")  # Always trigger warnings.

            # Calling `len` will create new generators each time.
            tractogram = LazyTractogram(streamlines,
                                        data_per_streamline=data_per_streamline,
                                        data_per_point=data_per_point)
            assert_true(tractogram._nb_streamlines is None)

            # This should produce a warning message.
            assert_equal(len(tractogram), self.nb_streamlines)
            assert_equal(tractogram._nb_streamlines, self.nb_streamlines)
            assert_equal(len(w), 1)

            tractogram = LazyTractogram(streamlines,
                                        data_per_streamline=data_per_streamline,
                                        data_per_point=data_per_point)

            # New instances should still produce a warning message.
            assert_equal(len(tractogram), self.nb_streamlines)
            assert_equal(len(w), 2)
            assert_true(issubclass(w[-1].category, UsageWarning))

            # Calling again 'len' again should *not* produce a warning.
            assert_equal(len(tractogram), self.nb_streamlines)
            assert_equal(len(w), 2)

        with clear_and_catch_warnings(record=True, modules=modules) as w:
            # Once we iterated through the tractogram, we know the length.

            tractogram = LazyTractogram(streamlines,
                                        data_per_streamline=data_per_streamline,
                                        data_per_point=data_per_point)

            assert_true(tractogram._nb_streamlines is None)
            isiterable(tractogram)  # Force to iterate through all streamlines.
            assert_equal(tractogram._nb_streamlines, len(self.streamlines))
            # This should *not* produce a warning.
            assert_equal(len(tractogram), len(self.streamlines))
            assert_equal(len(w), 0)

    def test_lazy_tractogram_apply_affine(self):
        streamlines = lambda: (x for x in self.streamlines)
        data_per_point = {"colors": self.colors_func}
        data_per_streamline = {'mean_curv': self.mean_curvature_func,
                               'mean_color': self.mean_color_func}

        affine = np.eye(4)
        scaling = np.array((1, 2, 3), dtype=float)
        affine[range(3), range(3)] = scaling

        tractogram = LazyTractogram(streamlines,
                                    data_per_streamline=data_per_streamline,
                                    data_per_point=data_per_point)

        tractogram.apply_affine(affine)
        assert_true(isiterable(tractogram))
        assert_equal(len(tractogram), len(self.streamlines))
        for s1, s2 in zip(tractogram.streamlines, self.streamlines):
            assert_array_almost_equal(s1, s2*scaling)
