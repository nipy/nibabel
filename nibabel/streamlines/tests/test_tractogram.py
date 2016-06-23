import sys
import unittest
import numpy as np
import warnings

from nibabel.testing import assert_arrays_equal
from nibabel.testing import clear_and_catch_warnings
from nose.tools import assert_equal, assert_raises, assert_true
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nibabel.externals.six.moves import zip

from .. import tractogram as module_tractogram
from ..tractogram import TractogramItem, Tractogram, LazyTractogram
from ..tractogram import PerArrayDict, PerArraySequenceDict, LazyDict

DATA = {}


def setup():
    global DATA
    DATA['rng'] = np.random.RandomState(1234)
    DATA['streamlines'] = [np.arange(1*3, dtype="f4").reshape((1, 3)),
                           np.arange(2*3, dtype="f4").reshape((2, 3)),
                           np.arange(5*3, dtype="f4").reshape((5, 3))]

    DATA['fa'] = [np.array([[0.2]], dtype="f4"),
                  np.array([[0.3],
                            [0.4]], dtype="f4"),
                  np.array([[0.5],
                            [0.6],
                            [0.6],
                            [0.7],
                            [0.8]], dtype="f4")]

    DATA['colors'] = [np.array([(1, 0, 0)]*1, dtype="f4"),
                      np.array([(0, 1, 0)]*2, dtype="f4"),
                      np.array([(0, 0, 1)]*5, dtype="f4")]

    DATA['mean_curvature'] = [np.array([1.11], dtype="f4"),
                              np.array([2.11], dtype="f4"),
                              np.array([3.11], dtype="f4")]

    DATA['mean_torsion'] = [np.array([1.22], dtype="f4"),
                            np.array([2.22], dtype="f4"),
                            np.array([3.22], dtype="f4")]

    DATA['mean_colors'] = [np.array([1, 0, 0], dtype="f4"),
                           np.array([0, 1, 0], dtype="f4"),
                           np.array([0, 0, 1], dtype="f4")]

    DATA['data_per_point'] = {'colors': DATA['colors'],
                              'fa': DATA['fa']}
    DATA['data_per_streamline'] = {'mean_curvature': DATA['mean_curvature'],
                                   'mean_torsion': DATA['mean_torsion'],
                                   'mean_colors': DATA['mean_colors']}

    DATA['empty_tractogram'] = Tractogram(affine_to_rasmm=np.eye(4))
    DATA['simple_tractogram'] = Tractogram(DATA['streamlines'],
                                           affine_to_rasmm=np.eye(4))
    DATA['tractogram'] = Tractogram(DATA['streamlines'],
                                    DATA['data_per_streamline'],
                                    DATA['data_per_point'],
                                    affine_to_rasmm=np.eye(4))

    DATA['streamlines_func'] = lambda: (e for e in DATA['streamlines'])
    fa_func = lambda: (e for e in DATA['fa'])
    colors_func = lambda: (e for e in DATA['colors'])
    mean_curvature_func = lambda: (e for e in DATA['mean_curvature'])
    mean_torsion_func = lambda: (e for e in DATA['mean_torsion'])
    mean_colors_func = lambda: (e for e in DATA['mean_colors'])

    DATA['data_per_point_func'] = {'colors': colors_func,
                                   'fa': fa_func}
    DATA['data_per_streamline_func'] = {'mean_curvature': mean_curvature_func,
                                        'mean_torsion': mean_torsion_func,
                                        'mean_colors': mean_colors_func}

    DATA['lazy_tractogram'] = LazyTractogram(DATA['streamlines_func'],
                                             DATA['data_per_streamline_func'],
                                             DATA['data_per_point_func'],
                                             affine_to_rasmm=np.eye(4))


def check_tractogram_item(tractogram_item,
                          streamline,
                          data_for_streamline={},
                          data_for_points={}):

    assert_array_equal(tractogram_item.streamline, streamline)

    assert_equal(len(tractogram_item.data_for_streamline),
                 len(data_for_streamline))
    for key in data_for_streamline.keys():
        assert_array_equal(tractogram_item.data_for_streamline[key],
                           data_for_streamline[key])

    assert_equal(len(tractogram_item.data_for_points), len(data_for_points))
    for key in data_for_points.keys():
        assert_arrays_equal(tractogram_item.data_for_points[key],
                            data_for_points[key])


def assert_tractogram_item_equal(t1, t2):
    check_tractogram_item(t1, t2.streamline,
                          t2.data_for_streamline, t2.data_for_points)


def check_tractogram(tractogram,
                     streamlines=[],
                     data_per_streamline={},
                     data_per_point={}):
    streamlines = list(streamlines)
    assert_equal(len(tractogram), len(streamlines))
    assert_arrays_equal(tractogram.streamlines, streamlines)
    [t for t in tractogram]  # Force iteration through tractogram.

    assert_equal(len(tractogram.data_per_streamline), len(data_per_streamline))
    for key in data_per_streamline.keys():
        assert_arrays_equal(tractogram.data_per_streamline[key],
                            data_per_streamline[key])

    assert_equal(len(tractogram.data_per_point), len(data_per_point))
    for key in data_per_point.keys():
        assert_arrays_equal(tractogram.data_per_point[key],
                            data_per_point[key])


def assert_tractogram_equal(t1, t2):
    check_tractogram(t1, t2.streamlines,
                     t2.data_per_streamline, t2.data_per_point)


class TestPerArrayDict(unittest.TestCase):

    def test_per_array_dict_creation(self):
        # Create a PerArrayDict object using another
        # PerArrayDict object.
        nb_streamlines = len(DATA['tractogram'])
        data_per_streamline = DATA['tractogram'].data_per_streamline
        data_dict = PerArrayDict(nb_streamlines, data_per_streamline)
        assert_equal(data_dict.keys(), data_per_streamline.keys())
        for k in data_dict.keys():
            assert_array_equal(data_dict[k], data_per_streamline[k])

        del data_dict['mean_curvature']
        assert_equal(len(data_dict),
                     len(data_per_streamline)-1)

        # Create a PerArrayDict object using an existing dict object.
        data_per_streamline = DATA['data_per_streamline']
        data_dict = PerArrayDict(nb_streamlines, data_per_streamline)
        assert_equal(data_dict.keys(), data_per_streamline.keys())
        for k in data_dict.keys():
            assert_array_equal(data_dict[k], data_per_streamline[k])

        del data_dict['mean_curvature']
        assert_equal(len(data_dict), len(data_per_streamline)-1)

        # Create a PerArrayDict object using keyword arguments.
        data_per_streamline = DATA['data_per_streamline']
        data_dict = PerArrayDict(nb_streamlines, **data_per_streamline)
        assert_equal(data_dict.keys(), data_per_streamline.keys())
        for k in data_dict.keys():
            assert_array_equal(data_dict[k], data_per_streamline[k])

        del data_dict['mean_curvature']
        assert_equal(len(data_dict), len(data_per_streamline)-1)

    def test_getitem(self):
        sdict = PerArrayDict(len(DATA['tractogram']),
                             DATA['data_per_streamline'])

        assert_raises(KeyError, sdict.__getitem__, 'invalid')

        # Test slicing and advanced indexing.
        for k, v in DATA['tractogram'].data_per_streamline.items():
            assert_true(k in sdict)
            assert_arrays_equal(sdict[k], v)
            assert_arrays_equal(sdict[::2][k], v[::2])
            assert_arrays_equal(sdict[::-1][k], v[::-1])
            assert_arrays_equal(sdict[-1][k], v[-1])
            assert_arrays_equal(sdict[[0, -1]][k], v[[0, -1]])


class TestPerArraySequenceDict(unittest.TestCase):

    def test_per_array_sequence_dict_creation(self):
        # Create a PerArraySequenceDict object using another
        # PerArraySequenceDict object.
        total_nb_rows = DATA['tractogram'].streamlines.total_nb_rows
        data_per_point = DATA['tractogram'].data_per_point
        data_dict = PerArraySequenceDict(total_nb_rows, data_per_point)
        assert_equal(data_dict.keys(), data_per_point.keys())
        for k in data_dict.keys():
            assert_arrays_equal(data_dict[k], data_per_point[k])

        del data_dict['fa']
        assert_equal(len(data_dict),
                     len(data_per_point)-1)

        # Create a PerArraySequenceDict object using an existing dict object.
        data_per_point = DATA['data_per_point']
        data_dict = PerArraySequenceDict(total_nb_rows, data_per_point)
        assert_equal(data_dict.keys(), data_per_point.keys())
        for k in data_dict.keys():
            assert_arrays_equal(data_dict[k], data_per_point[k])

        del data_dict['fa']
        assert_equal(len(data_dict), len(data_per_point)-1)

        # Create a PerArraySequenceDict object using keyword arguments.
        data_per_point = DATA['data_per_point']
        data_dict = PerArraySequenceDict(total_nb_rows, **data_per_point)
        assert_equal(data_dict.keys(), data_per_point.keys())
        for k in data_dict.keys():
            assert_arrays_equal(data_dict[k], data_per_point[k])

        del data_dict['fa']
        assert_equal(len(data_dict), len(data_per_point)-1)

    def test_getitem(self):
        total_nb_rows = DATA['tractogram'].streamlines.total_nb_rows
        sdict = PerArraySequenceDict(total_nb_rows, DATA['data_per_point'])

        assert_raises(KeyError, sdict.__getitem__, 'invalid')

        # Test slicing and advanced indexing.
        for k, v in DATA['tractogram'].data_per_point.items():
            assert_true(k in sdict)
            assert_arrays_equal(sdict[k], v)
            assert_arrays_equal(sdict[::2][k], v[::2])
            assert_arrays_equal(sdict[::-1][k], v[::-1])
            assert_arrays_equal(sdict[-1][k], v[-1])
            assert_arrays_equal(sdict[[0, -1]][k], v[[0, -1]])


class TestLazyDict(unittest.TestCase):

    def test_lazydict_creation(self):
        data_dict = LazyDict(DATA['data_per_streamline_func'])
        assert_equal(data_dict.keys(), DATA['data_per_streamline_func'].keys())
        for k in data_dict.keys():
            assert_array_equal(list(data_dict[k]),
                               list(DATA['data_per_streamline'][k]))

        assert_equal(len(data_dict),
                     len(DATA['data_per_streamline_func']))


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

    def test_tractogram_creation(self):
        # Create an empty tractogram.
        tractogram = Tractogram()
        check_tractogram(tractogram)
        assert_true(tractogram.affine_to_rasmm is None)

        # Create a tractogram with only streamlines
        tractogram = Tractogram(streamlines=DATA['streamlines'])
        check_tractogram(tractogram, DATA['streamlines'])

        # Create a tractogram with a given affine_to_rasmm.
        affine = np.diag([1, 2, 3, 1])
        tractogram = Tractogram(affine_to_rasmm=affine)
        assert_array_equal(tractogram.affine_to_rasmm, affine)

        # Create a tractogram with streamlines and other data.
        tractogram = Tractogram(DATA['streamlines'],
                                DATA['data_per_streamline'],
                                DATA['data_per_point'])

        check_tractogram(tractogram,
                         DATA['streamlines'],
                         DATA['data_per_streamline'],
                         DATA['data_per_point'])

        # Create a tractogram from another tractogram attributes.
        tractogram2 = Tractogram(tractogram.streamlines,
                                 tractogram.data_per_streamline,
                                 tractogram.data_per_point)

        assert_tractogram_equal(tractogram2, tractogram)

        # Create a tractogram from a LazyTractogram object.
        tractogram = LazyTractogram(DATA['streamlines_func'],
                                    DATA['data_per_streamline_func'],
                                    DATA['data_per_point_func'])

        tractogram2 = Tractogram(tractogram.streamlines,
                                 tractogram.data_per_streamline,
                                 tractogram.data_per_point)

        # Inconsistent number of scalars between streamlines
        wrong_data = [[(1, 0, 0)]*1,
                      [(0, 1, 0), (0, 1)],
                      [(0, 0, 1)]*5]

        data_per_point = {'wrong_data': wrong_data}
        assert_raises(ValueError, Tractogram, DATA['streamlines'],
                      data_per_point=data_per_point)

        # Inconsistent number of scalars between streamlines
        wrong_data = [[(1, 0, 0)]*1,
                      [(0, 1)]*2,
                      [(0, 0, 1)]*5]

        data_per_point = {'wrong_data': wrong_data}
        assert_raises(ValueError, Tractogram, DATA['streamlines'],
                      data_per_point=data_per_point)

    def test_setting_affine_to_rasmm(self):
        tractogram = DATA['tractogram'].copy()
        affine = np.diag(range(4))

        # Test assigning None.
        tractogram.affine_to_rasmm = None
        assert_true(tractogram.affine_to_rasmm is None)

        # Test assigning a valid ndarray (should make a copy).
        tractogram.affine_to_rasmm = affine
        assert_true(tractogram.affine_to_rasmm is not affine)

        # Test assigning a list of lists.
        tractogram.affine_to_rasmm = affine.tolist()
        assert_array_equal(tractogram.affine_to_rasmm, affine)

        # Test assigning a ndarray with wrong shape.
        assert_raises(ValueError, setattr, tractogram,
                      "affine_to_rasmm", affine[::2])

    def test_tractogram_getitem(self):
        # Retrieve TractogramItem by their index.
        for i, t in enumerate(DATA['tractogram']):
            assert_tractogram_item_equal(DATA['tractogram'][i], t)

            if sys.version_info < (3,):
                assert_tractogram_item_equal(DATA['tractogram'][long(i)], t)

        # Get one TractogramItem out of two.
        tractogram_view = DATA['simple_tractogram'][::2]
        check_tractogram(tractogram_view, DATA['streamlines'][::2])

        # Use slicing.
        r_tractogram = DATA['tractogram'][::-1]
        check_tractogram(r_tractogram,
                         DATA['streamlines'][::-1],
                         DATA['tractogram'].data_per_streamline[::-1],
                         DATA['tractogram'].data_per_point[::-1])

        # Make sure slicing conserves the affine_to_rasmm property.
        tractogram = DATA['tractogram'].copy()
        tractogram.affine_to_rasmm = DATA['rng'].rand(4, 4)
        tractogram_view = tractogram[::2]
        assert_array_equal(tractogram_view.affine_to_rasmm,
                           tractogram.affine_to_rasmm)

    def test_tractogram_add_new_data(self):
        # Tractogram with only streamlines
        t = DATA['simple_tractogram'].copy()
        t.data_per_point['fa'] = DATA['fa']
        t.data_per_point['colors'] = DATA['colors']
        t.data_per_streamline['mean_curvature'] = DATA['mean_curvature']
        t.data_per_streamline['mean_torsion'] = DATA['mean_torsion']
        t.data_per_streamline['mean_colors'] = DATA['mean_colors']
        assert_tractogram_equal(t, DATA['tractogram'])

        # Retrieve tractogram by their index.
        for i, item in enumerate(t):
            assert_tractogram_item_equal(t[i], item)

        # Use slicing.
        r_tractogram = t[::-1]
        check_tractogram(r_tractogram,
                         t.streamlines[::-1],
                         t.data_per_streamline[::-1],
                         t.data_per_point[::-1])

        # Add new data to a tractogram for which its `streamlines` is a view.
        t = Tractogram(DATA['streamlines']*2, affine_to_rasmm=np.eye(4))
        t = t[:len(DATA['streamlines'])]  # Create a view of `streamlines`
        t.data_per_point['fa'] = DATA['fa']
        t.data_per_point['colors'] = DATA['colors']
        t.data_per_streamline['mean_curvature'] = DATA['mean_curvature']
        t.data_per_streamline['mean_torsion'] = DATA['mean_torsion']
        t.data_per_streamline['mean_colors'] = DATA['mean_colors']
        assert_tractogram_equal(t, DATA['tractogram'])

    def test_tractogram_copy(self):
        # Create a copy of a tractogram.
        tractogram = DATA['tractogram'].copy()

        # Check we copied the data and not simply created new references.
        assert_true(tractogram is not DATA['tractogram'])
        assert_true(tractogram.streamlines
                    is not DATA['tractogram'].streamlines)
        assert_true(tractogram.data_per_streamline
                    is not DATA['tractogram'].data_per_streamline)
        assert_true(tractogram.data_per_point
                    is not DATA['tractogram'].data_per_point)

        for key in tractogram.data_per_streamline:
            assert_true(tractogram.data_per_streamline[key]
                        is not DATA['tractogram'].data_per_streamline[key])

        for key in tractogram.data_per_point:
            assert_true(tractogram.data_per_point[key]
                        is not DATA['tractogram'].data_per_point[key])

        # Check the values of the data are the same.
        assert_tractogram_equal(tractogram, DATA['tractogram'])

    def test_creating_invalid_tractogram(self):
        # Not enough data_per_point for all the points of all streamlines.
        scalars = [[(1, 0, 0)]*1,
                   [(0, 1, 0)]*2,
                   [(0, 0, 1)]*3]  # Last streamlines has 5 points.

        assert_raises(ValueError, Tractogram, DATA['streamlines'],
                      data_per_point={'scalars': scalars})

        # Not enough data_per_streamline for all streamlines.
        properties = [np.array([1.11, 1.22], dtype="f4"),
                      np.array([3.11, 3.22], dtype="f4")]

        assert_raises(ValueError, Tractogram, DATA['streamlines'],
                      data_per_streamline={'properties': properties})

        # Inconsistent dimension for a data_per_point.
        scalars = [[(1, 0, 0)]*1,
                   [(0, 1)]*2,
                   [(0, 0, 1)]*5]

        assert_raises(ValueError, Tractogram, DATA['streamlines'],
                      data_per_point={'scalars': scalars})

        # Inconsistent dimension for a data_per_streamline.
        properties = [[1.11, 1.22],
                      [2.11],
                      [3.11, 3.22]]

        assert_raises(ValueError, Tractogram, DATA['streamlines'],
                      data_per_streamline={'properties': properties})

        # Too many dimension for a data_per_streamline.
        properties = [np.array([[1.11], [1.22]], dtype="f4"),
                      np.array([[2.11], [2.22]], dtype="f4"),
                      np.array([[3.11], [3.22]], dtype="f4")]

        assert_raises(ValueError, Tractogram, DATA['streamlines'],
                      data_per_streamline={'properties': properties})

    def test_tractogram_apply_affine(self):
        tractogram = DATA['tractogram'].copy()
        affine = np.eye(4)
        scaling = np.array((1, 2, 3), dtype=float)
        affine[range(3), range(3)] = scaling

        # Apply the affine to the streamline in a lazy manner.
        transformed_tractogram = tractogram.apply_affine(affine, lazy=True)
        assert_true(type(transformed_tractogram) is LazyTractogram)
        check_tractogram(transformed_tractogram,
                         streamlines=[s*scaling for s in DATA['streamlines']],
                         data_per_streamline=DATA['data_per_streamline'],
                         data_per_point=DATA['data_per_point'])
        assert_array_equal(transformed_tractogram.affine_to_rasmm,
                           np.dot(np.eye(4), np.linalg.inv(affine)))
        # Make sure streamlines of the original tractogram have not been
        # modified.
        assert_arrays_equal(tractogram.streamlines, DATA['streamlines'])

        # Apply the affine to the streamlines in-place.
        transformed_tractogram = tractogram.apply_affine(affine)
        assert_true(transformed_tractogram is tractogram)
        check_tractogram(tractogram,
                         streamlines=[s*scaling for s in DATA['streamlines']],
                         data_per_streamline=DATA['data_per_streamline'],
                         data_per_point=DATA['data_per_point'])

        # Apply affine again and check the affine_to_rasmm.
        transformed_tractogram = tractogram.apply_affine(affine)
        assert_array_equal(transformed_tractogram.affine_to_rasmm,
                           np.dot(np.eye(4), np.dot(np.linalg.inv(affine),
                                                    np.linalg.inv(affine))))

        # Check that applying an affine and its inverse give us back the
        # original streamlines.
        tractogram = DATA['tractogram'].copy()
        affine = np.random.RandomState(1234).randn(4, 4)
        affine[-1] = [0, 0, 0, 1]  # Remove perspective projection.

        tractogram.apply_affine(affine)
        tractogram.apply_affine(np.linalg.inv(affine))
        assert_array_almost_equal(tractogram.affine_to_rasmm, np.eye(4))
        for s1, s2 in zip(tractogram.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)

        # Test applying the identity transformation.
        tractogram = DATA['tractogram'].copy()
        tractogram.apply_affine(np.eye(4))
        for s1, s2 in zip(tractogram.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)

        # Test removing affine_to_rasmm
        tractogram = DATA['tractogram'].copy()
        tractogram.affine_to_rasmm = None
        tractogram.apply_affine(affine)
        assert_true(tractogram.affine_to_rasmm is None)

    def test_tractogram_to_world(self):
        tractogram = DATA['tractogram'].copy()
        affine = np.random.RandomState(1234).randn(4, 4)
        affine[-1] = [0, 0, 0, 1]  # Remove perspective projection.

        # Apply the affine to the streamlines, then bring them back
        # to world space in a lazy manner.
        transformed_tractogram = tractogram.apply_affine(affine)
        assert_array_equal(transformed_tractogram.affine_to_rasmm,
                           np.linalg.inv(affine))

        tractogram_world = transformed_tractogram.to_world(lazy=True)
        assert_true(type(tractogram_world) is LazyTractogram)
        assert_array_almost_equal(tractogram_world.affine_to_rasmm,
                                  np.eye(4))
        for s1, s2 in zip(tractogram_world.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)

        # Bring them back streamlines to world space in a in-place manner.
        tractogram_world = transformed_tractogram.to_world()
        assert_true(tractogram_world is tractogram)
        assert_array_almost_equal(tractogram.affine_to_rasmm, np.eye(4))
        for s1, s2 in zip(tractogram.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)

        # Calling to_world twice should do nothing.
        tractogram_world2 = transformed_tractogram.to_world()
        assert_true(tractogram_world2 is tractogram)
        assert_array_almost_equal(tractogram.affine_to_rasmm, np.eye(4))
        for s1, s2 in zip(tractogram.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)

        # Calling to_world when affine_to_rasmm is None should fail.
        tractogram = DATA['tractogram'].copy()
        tractogram.affine_to_rasmm = None
        assert_raises(ValueError, tractogram.to_world)


class TestLazyTractogram(unittest.TestCase):

    def test_lazy_tractogram_creation(self):
        # To create tractogram from arrays use `Tractogram`.
        assert_raises(TypeError, LazyTractogram, DATA['streamlines'])

        # Streamlines and other data as generators
        streamlines = (x for x in DATA['streamlines'])
        data_per_point = {"colors": (x for x in DATA['colors'])}
        data_per_streamline = {'mean_torsion': (x for x in DATA['mean_torsion']),
                               'mean_colors': (x for x in DATA['mean_colors'])}

        # Creating LazyTractogram with generators is not allowed as
        # generators get exhausted and are not reusable unlike generator function.
        assert_raises(TypeError, LazyTractogram, streamlines)
        assert_raises(TypeError, LazyTractogram,
                      data_per_streamline=data_per_streamline)
        assert_raises(TypeError, LazyTractogram, DATA['streamlines'],
                      data_per_point=data_per_point)

        # Empty `LazyTractogram`
        tractogram = LazyTractogram()
        check_tractogram(tractogram)
        assert_true(tractogram.affine_to_rasmm is None)

        # Create tractogram with streamlines and other data
        tractogram = LazyTractogram(DATA['streamlines_func'],
                                    DATA['data_per_streamline_func'],
                                    DATA['data_per_point_func'])

        [t for t in tractogram]  # Force iteration through tractogram.
        assert_equal(len(tractogram), len(DATA['streamlines']))

        # Generator functions get re-called and creates new iterators.
        for i in range(2):
            assert_tractogram_equal(tractogram, DATA['tractogram'])

    def test_lazy_tractogram_from_data_func(self):
        # Create an empty `LazyTractogram` yielding nothing.
        _empty_data_gen = lambda: iter([])

        tractogram = LazyTractogram.from_data_func(_empty_data_gen)
        check_tractogram(tractogram)

        # Create `LazyTractogram` from a generator function yielding TractogramItem.
        data = [DATA['streamlines'], DATA['fa'], DATA['colors'],
                DATA['mean_curvature'], DATA['mean_torsion'],
                DATA['mean_colors']]

        def _data_gen():
            for d in zip(*data):
                data_for_points = {'fa': d[1],
                                   'colors': d[2]}
                data_for_streamline = {'mean_curvature': d[3],
                                       'mean_torsion': d[4],
                                       'mean_colors': d[5]}
                yield TractogramItem(d[0],
                                     data_for_streamline,
                                     data_for_points)

        tractogram = LazyTractogram.from_data_func(_data_gen)
        assert_tractogram_equal(tractogram, DATA['tractogram'])

        # Creating a LazyTractogram from not a corouting should raise an error.
        assert_raises(TypeError, LazyTractogram.from_data_func, _data_gen())

    def test_lazy_tractogram_getitem(self):
        assert_raises(NotImplementedError,
                      DATA['lazy_tractogram'].__getitem__, 0)

    def test_lazy_tractogram_len(self):
        modules = [module_tractogram]  # Modules for which to catch warnings.
        with clear_and_catch_warnings(record=True, modules=modules) as w:
            warnings.simplefilter("always")  # Always trigger warnings.

            # Calling `len` will create new generators each time.
            tractogram = LazyTractogram(DATA['streamlines_func'])
            assert_true(tractogram._nb_streamlines is None)

            # This should produce a warning message.
            assert_equal(len(tractogram), len(DATA['streamlines']))
            assert_equal(tractogram._nb_streamlines, len(DATA['streamlines']))
            assert_equal(len(w), 1)

            tractogram = LazyTractogram(DATA['streamlines_func'])

            # New instances should still produce a warning message.
            assert_equal(len(tractogram), len(DATA['streamlines']))
            assert_equal(len(w), 2)
            assert_true(issubclass(w[-1].category, Warning))

            # Calling again 'len' again should *not* produce a warning.
            assert_equal(len(tractogram), len(DATA['streamlines']))
            assert_equal(len(w), 2)

        with clear_and_catch_warnings(record=True, modules=modules) as w:
            # Once we iterated through the tractogram, we know the length.

            tractogram = LazyTractogram(DATA['streamlines_func'])

            assert_true(tractogram._nb_streamlines is None)
            [t for t in tractogram]  # Force iteration through tractogram.
            assert_equal(tractogram._nb_streamlines, len(DATA['streamlines']))
            # This should *not* produce a warning.
            assert_equal(len(tractogram), len(DATA['streamlines']))
            assert_equal(len(w), 0)

    def test_lazy_tractogram_apply_affine(self):
        affine = np.eye(4)
        scaling = np.array((1, 2, 3), dtype=float)
        affine[range(3), range(3)] = scaling

        tractogram = DATA['lazy_tractogram'].copy()

        transformed_tractogram = tractogram.apply_affine(affine)
        assert_true(transformed_tractogram is not tractogram)
        assert_array_equal(tractogram._affine_to_apply, np.eye(4))
        assert_array_equal(tractogram.affine_to_rasmm, np.eye(4))
        assert_array_equal(transformed_tractogram._affine_to_apply, affine)
        assert_array_equal(transformed_tractogram.affine_to_rasmm,
                           np.dot(np.eye(4), np.linalg.inv(affine)))
        check_tractogram(transformed_tractogram,
                         streamlines=[s*scaling for s in DATA['streamlines']],
                         data_per_streamline=DATA['data_per_streamline'],
                         data_per_point=DATA['data_per_point'])

        # Apply affine again and check the affine_to_rasmm.
        transformed_tractogram = transformed_tractogram.apply_affine(affine)
        assert_array_equal(transformed_tractogram._affine_to_apply,
                           np.dot(affine, affine))
        assert_array_equal(transformed_tractogram.affine_to_rasmm,
                           np.dot(np.eye(4), np.dot(np.linalg.inv(affine),
                                                    np.linalg.inv(affine))))

        # Calling to_world when affine_to_rasmm is None should fail.
        tractogram = DATA['lazy_tractogram'].copy()
        tractogram.affine_to_rasmm = None
        assert_raises(ValueError, tractogram.to_world)

    def test_tractogram_to_world(self):
        tractogram = DATA['lazy_tractogram'].copy()
        affine = np.random.RandomState(1234).randn(4, 4)
        affine[-1] = [0, 0, 0, 1]  # Remove perspective projection.

        # Apply the affine to the streamlines, then bring them back
        # to world space in a lazy manner.
        transformed_tractogram = tractogram.apply_affine(affine)
        assert_array_equal(transformed_tractogram.affine_to_rasmm,
                           np.linalg.inv(affine))

        tractogram_world = transformed_tractogram.to_world()
        assert_true(tractogram_world is not transformed_tractogram)
        assert_array_almost_equal(tractogram_world.affine_to_rasmm,
                                  np.eye(4))
        for s1, s2 in zip(tractogram_world.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)

        # Calling to_world twice should do nothing.
        tractogram_world = tractogram_world.to_world()
        assert_array_almost_equal(tractogram_world.affine_to_rasmm, np.eye(4))
        for s1, s2 in zip(tractogram_world.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)

        # Calling to_world when affine_to_rasmm is None should fail.
        tractogram = DATA['lazy_tractogram'].copy()
        tractogram.affine_to_rasmm = None
        assert_raises(ValueError, tractogram.to_world)

    def test_lazy_tractogram_copy(self):
        # Create a copy of the lazy tractogram.
        tractogram = DATA['lazy_tractogram'].copy()

        # Check we copied the data and not simply created new references.
        assert_true(tractogram is not DATA['lazy_tractogram'])

        # When copying LazyTractogram, the generator function yielding streamlines
        # should stay the same.
        assert_true(tractogram._streamlines
                    is DATA['lazy_tractogram']._streamlines)

        # Copying LazyTractogram, creates new internal LazyDict objects,
        # but generator functions contained in it should stay the same.
        assert_true(tractogram._data_per_streamline
                    is not DATA['lazy_tractogram']._data_per_streamline)
        assert_true(tractogram._data_per_point
                    is not DATA['lazy_tractogram']._data_per_point)

        for key in tractogram.data_per_streamline:
            assert_true(tractogram.data_per_streamline.store[key]
                        is DATA['lazy_tractogram'].data_per_streamline.store[key])

        for key in tractogram.data_per_point:
            assert_true(tractogram.data_per_point.store[key]
                        is DATA['lazy_tractogram'].data_per_point.store[key])

        # The affine should be a copy.
        assert_true(tractogram._affine_to_apply
                    is not DATA['lazy_tractogram']._affine_to_apply)
        assert_array_equal(tractogram._affine_to_apply,
                           DATA['lazy_tractogram']._affine_to_apply)

        # Check the data are the equivalent.
        assert_tractogram_equal(tractogram, DATA['tractogram'])
