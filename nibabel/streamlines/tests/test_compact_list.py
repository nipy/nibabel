import os
import unittest
import tempfile
import numpy as np

from nose.tools import assert_equal, assert_raises, assert_true
from nibabel.testing import assert_arrays_equal
from numpy.testing import assert_array_equal
from nibabel.externals.six.moves import zip, zip_longest

from ..compact_list import CompactList


class TestCompactList(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(42)
        self.data = [rng.rand(rng.randint(10, 50), 3) for _ in range(10)]
        self.lengths = list(map(len, self.data))
        self.clist = CompactList(self.data)

    def test_creating_empty_compactlist(self):
        clist = CompactList()
        assert_equal(len(clist), 0)
        assert_equal(len(clist._offsets), 0)
        assert_equal(len(clist._lengths), 0)
        assert_equal(clist._data.ndim, 0)
        assert_true(clist.common_shape == ())

    def test_creating_compactlist_from_list(self):
        rng = np.random.RandomState(42)
        data = [rng.rand(rng.randint(10, 50), 3) for _ in range(10)]
        lengths = list(map(len, data))

        clist = CompactList(data)
        assert_equal(len(clist), len(data))
        assert_equal(len(clist._offsets), len(data))
        assert_equal(len(clist._lengths), len(data))
        assert_equal(clist._data.shape[0], sum(lengths))
        assert_equal(clist._data.shape[1], 3)
        assert_array_equal(clist._offsets, np.r_[0, np.cumsum(lengths)[:-1]])
        assert_array_equal(clist._lengths, lengths)
        assert_equal(clist.common_shape, data[0].shape[1:])

        # Empty list
        clist = CompactList([])
        assert_equal(len(clist), 0)
        assert_equal(len(clist._offsets), 0)
        assert_equal(len(clist._lengths), 0)
        assert_equal(clist._data.ndim, 0)
        assert_true(clist.common_shape == ())

        # Force CompactList constructor to use buffering.
        old_buffer_size = CompactList.BUFFER_SIZE
        CompactList.BUFFER_SIZE = 1
        clist = CompactList(data)
        assert_equal(len(clist), len(data))
        assert_equal(len(clist._offsets), len(data))
        assert_equal(len(clist._lengths), len(data))
        assert_equal(clist._data.shape[0], sum(lengths))
        assert_equal(clist._data.shape[1], 3)
        assert_array_equal(clist._offsets, np.r_[0, np.cumsum(lengths)[:-1]])
        assert_array_equal(clist._lengths, lengths)
        assert_equal(clist.common_shape, data[0].shape[1:])
        CompactList.BUFFER_SIZE = old_buffer_size

    def test_creating_compactlist_from_generator(self):
        rng = np.random.RandomState(42)
        data = [rng.rand(rng.randint(10, 50), 3) for _ in range(10)]
        lengths = list(map(len, data))

        gen = (e for e in data)
        clist = CompactList(gen)
        assert_equal(len(clist), len(data))
        assert_equal(len(clist._offsets), len(data))
        assert_equal(len(clist._lengths), len(data))
        assert_equal(clist._data.shape[0], sum(lengths))
        assert_equal(clist._data.shape[1], 3)
        assert_array_equal(clist._offsets, np.r_[0, np.cumsum(lengths)[:-1]])
        assert_array_equal(clist._lengths, lengths)
        assert_equal(clist.common_shape, data[0].shape[1:])

        # Already consumed generator
        clist = CompactList(gen)
        assert_equal(len(clist), 0)
        assert_equal(len(clist._offsets), 0)
        assert_equal(len(clist._lengths), 0)
        assert_equal(clist._data.ndim, 0)
        assert_true(clist.common_shape == ())

    def test_creating_compactlist_from_compact_list(self):
        rng = np.random.RandomState(42)
        data = [rng.rand(rng.randint(10, 50), 3) for _ in range(10)]
        lengths = list(map(len, data))

        clist = CompactList(data)
        clist2 = CompactList(clist)
        assert_equal(len(clist2), len(data))
        assert_equal(len(clist2._offsets), len(data))
        assert_equal(len(clist2._lengths), len(data))
        assert_equal(clist2._data.shape[0], sum(lengths))
        assert_equal(clist2._data.shape[1], 3)
        assert_array_equal(clist2._offsets, np.r_[0, np.cumsum(lengths)[:-1]])
        assert_array_equal(clist2._lengths, lengths)
        assert_equal(clist2.common_shape, data[0].shape[1:])

    def test_compactlist_iter(self):
        for e, d in zip(self.clist, self.data):
            assert_array_equal(e, d)

        # Try iterate through a corrupted CompactList object.
        clist = self.clist.copy()
        clist._lengths = clist._lengths[::2]
        assert_raises(ValueError, list, clist)

    def test_compactlist_copy(self):
        clist = self.clist.copy()
        assert_array_equal(clist._data, self.clist._data)
        assert_true(clist._data is not self.clist._data)
        assert_array_equal(clist._offsets, self.clist._offsets)
        assert_true(clist._offsets is not self.clist._offsets)
        assert_array_equal(clist._lengths, self.clist._lengths)
        assert_true(clist._lengths is not self.clist._lengths)

        assert_equal(clist.common_shape, self.clist.common_shape)

        # When taking a copy of a `CompactList` generated by slicing.
        # Only needed data should be kept.
        clist = self.clist[::2].copy()

        assert_true(clist._data.shape[0] < self.clist._data.shape[0])
        assert_true(len(clist) < len(self.clist))
        assert_true(clist._data is not self.clist._data)
        assert_array_equal(clist._lengths, self.clist[::2]._lengths)
        assert_array_equal(clist._offsets,
                           np.cumsum(np.r_[0, self.clist[::2]._lengths])[:-1])
        assert_arrays_equal(clist, self.clist[::2])

    def test_compactlist_append(self):
        # Maybe not necessary if `self.setUp` is always called before a
        # test method, anyways create a copy just in case.
        clist = self.clist.copy()

        rng = np.random.RandomState(1234)
        element = rng.rand(rng.randint(10, 50), *self.clist.common_shape)
        clist.append(element)
        assert_equal(len(clist), len(self.clist)+1)
        assert_equal(clist._offsets[-1], len(self.clist._data))
        assert_equal(clist._lengths[-1], len(element))
        assert_array_equal(clist._data[-len(element):], element)

        # Append with different shape.
        element = rng.rand(rng.randint(10, 50), 42)
        assert_raises(ValueError, clist.append, element)

        # Append to an empty CompactList.
        clist = CompactList()
        rng = np.random.RandomState(1234)
        shape = (2, 3, 4)
        element = rng.rand(rng.randint(10, 50), *shape)
        clist.append(element)

        assert_equal(len(clist), 1)
        assert_equal(clist._offsets[-1], 0)
        assert_equal(clist._lengths[-1], len(element))
        assert_array_equal(clist._data, element)
        assert_equal(clist.common_shape, shape)

    def test_compactlist_extend(self):
        # Maybe not necessary if `self.setUp` is always called before a
        # test method, anyways create a copy just in case.
        clist = self.clist.copy()

        rng = np.random.RandomState(1234)
        shape = self.clist.common_shape
        new_data = [rng.rand(rng.randint(10, 50), *shape) for _ in range(10)]
        lengths = list(map(len, new_data))
        clist.extend(new_data)
        assert_equal(len(clist), len(self.clist)+len(new_data))
        assert_array_equal(clist._offsets[-len(new_data):],
                           len(self.clist._data) + np.cumsum([0] + lengths[:-1]))

        assert_array_equal(clist._lengths[-len(new_data):], lengths)
        assert_array_equal(clist._data[-sum(lengths):],
                           np.concatenate(new_data, axis=0))

        # Extend with another `CompactList` object.
        clist = self.clist.copy()
        new_clist = CompactList(new_data)
        clist.extend(new_clist)
        assert_equal(len(clist), len(self.clist)+len(new_clist))
        assert_array_equal(clist._offsets[-len(new_clist):],
                           len(self.clist._data) + np.cumsum(np.r_[0, lengths[:-1]]))

        assert_array_equal(clist._lengths[-len(new_clist):], lengths)
        assert_array_equal(clist._data[-sum(lengths):], new_clist._data)

        # Extend with another `CompactList` object that is a view (e.g. been sliced).
        # Need to make sure we extend only the data we need.
        clist = self.clist.copy()
        new_clist = CompactList(new_data)[::2]
        clist.extend(new_clist)
        assert_equal(len(clist), len(self.clist)+len(new_clist))
        assert_equal(len(clist._data), len(self.clist._data)+sum(new_clist._lengths))
        assert_array_equal(clist._offsets[-len(new_clist):],
                           len(self.clist._data) + np.cumsum(np.r_[0, new_clist._lengths[:-1]]))

        assert_array_equal(clist._lengths[-len(new_clist):], lengths[::2])
        assert_array_equal(clist._data[-sum(new_clist._lengths):], new_clist.copy()._data)
        assert_arrays_equal(clist[-len(new_clist):], new_clist)

        # Test extending an empty CompactList
        clist = CompactList()
        new_clist = CompactList(new_data)
        clist.extend(new_clist)
        assert_equal(len(clist), len(new_clist))
        assert_array_equal(clist._offsets, new_clist._offsets)
        assert_array_equal(clist._lengths, new_clist._lengths)
        assert_array_equal(clist._data, new_clist._data)

    def test_compactlist_getitem(self):
        # Get one item
        for i, e in enumerate(self.clist):
            assert_array_equal(self.clist[i], e)

        # Get multiple items (this will create a view).
        indices = list(range(len(self.clist)))
        clist_view = self.clist[indices]
        assert_true(clist_view is not self.clist)
        assert_true(clist_view._data is self.clist._data)
        assert_true(clist_view._offsets is not self.clist._offsets)
        assert_true(clist_view._lengths is not self.clist._lengths)
        assert_array_equal(clist_view._offsets, self.clist._offsets)
        assert_array_equal(clist_view._lengths, self.clist._lengths)
        assert_arrays_equal(clist_view, self.clist)

        # Get multiple items using ndarray of data type.
        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            clist_view = self.clist[np.array(indices, dtype=dtype)]
            assert_true(clist_view is not self.clist)
            assert_true(clist_view._data is self.clist._data)
            assert_true(clist_view._offsets is not self.clist._offsets)
            assert_true(clist_view._lengths is not self.clist._lengths)
            assert_array_equal(clist_view._offsets, self.clist._offsets)
            assert_array_equal(clist_view._lengths, self.clist._lengths)
            for e1, e2 in zip_longest(clist_view, self.clist):
                assert_array_equal(e1, e2)

        # Get slice (this will create a view).
        clist_view = self.clist[::2]
        assert_true(clist_view is not self.clist)
        assert_true(clist_view._data is self.clist._data)
        assert_array_equal(clist_view._offsets, self.clist._offsets[::2])
        assert_array_equal(clist_view._lengths, self.clist._lengths[::2])
        for i, e in enumerate(clist_view):
            assert_array_equal(e, self.clist[i*2])

        # Use advance indexing with ndarray of data type bool.
        idx = np.array([False, True, True, False, True])
        clist_view = self.clist[idx]
        assert_true(clist_view is not self.clist)
        assert_true(clist_view._data is self.clist._data)
        assert_array_equal(clist_view._offsets,
                           self.clist._offsets[idx])
        assert_array_equal(clist_view._lengths,
                           self.clist._lengths[idx])
        assert_array_equal(clist_view[0], self.clist[1])
        assert_array_equal(clist_view[1], self.clist[2])
        assert_array_equal(clist_view[2], self.clist[4])

        # Test invalid indexing
        assert_raises(TypeError, self.clist.__getitem__, 'abc')

    def test_compactlist_repr(self):
        # Test that calling repr on a CompactList object is not falling.
        repr(self.clist)

    def test_save_and_load_compact_list(self):

        # Test saving and loading an empty CompactList.
        with tempfile.TemporaryFile(mode="w+b", suffix=".npz") as f:
            clist = CompactList()
            clist.save(f)
            f.seek(0, os.SEEK_SET)
            loaded_clist = CompactList.from_filename(f)
            assert_array_equal(loaded_clist._data, clist._data)
            assert_array_equal(loaded_clist._offsets, clist._offsets)
            assert_array_equal(loaded_clist._lengths, clist._lengths)

        # Test saving and loading a CompactList.
        with tempfile.TemporaryFile(mode="w+b", suffix=".npz") as f:
            rng = np.random.RandomState(42)
            data = [rng.rand(rng.randint(10, 50), 3) for _ in range(10)]
            clist = CompactList(data)
            clist.save(f)
            f.seek(0, os.SEEK_SET)
            loaded_clist = CompactList.from_filename(f)
            assert_array_equal(loaded_clist._data, clist._data)
            assert_array_equal(loaded_clist._offsets, clist._offsets)
            assert_array_equal(loaded_clist._lengths, clist._lengths)
