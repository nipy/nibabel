import os
import unittest
import tempfile
import numpy as np

from nose.tools import assert_equal, assert_raises, assert_true
from nibabel.testing import assert_arrays_equal
from numpy.testing import assert_array_equal
from nibabel.externals.six.moves import zip, zip_longest

from ..array_sequence import ArraySequence


class TestArraySequence(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(42)
        self.data = [rng.rand(rng.randint(10, 50), 3) for _ in range(10)]
        self.lengths = list(map(len, self.data))
        self.seq = ArraySequence(self.data)

    def test_creating_empty_arraysequence(self):
        seq = ArraySequence()
        assert_equal(len(seq), 0)
        assert_equal(len(seq._offsets), 0)
        assert_equal(len(seq._lengths), 0)
        assert_equal(seq._data.ndim, 0)
        assert_true(seq.common_shape == ())

    def test_creating_arraysequence_from_list(self):
        rng = np.random.RandomState(42)

        # Empty list
        seq = ArraySequence([])
        assert_equal(len(seq), 0)
        assert_equal(len(seq._offsets), 0)
        assert_equal(len(seq._lengths), 0)
        assert_equal(seq._data.ndim, 0)
        assert_true(seq.common_shape == ())

        # List of ndarrays.
        N = 5
        nb_arrays = 10
        for ndim in range(0, N+1):
            common_shape = tuple([rng.randint(1, 10) for _ in range(ndim-1)])
            data = [rng.rand(*(rng.randint(10, 50),) + common_shape)
                    for _ in range(nb_arrays)]
            lengths = list(map(len, data))

            seq = ArraySequence(data)
            assert_equal(len(seq), len(data))
            assert_equal(len(seq), nb_arrays)
            assert_equal(len(seq._offsets), nb_arrays)
            assert_equal(len(seq._lengths), nb_arrays)
            assert_equal(seq._data.shape[0], sum(lengths))
            assert_equal(seq._data.shape[1:], common_shape)
            assert_equal(seq.common_shape, common_shape)
            assert_array_equal(seq._offsets, np.r_[0, np.cumsum(lengths)[:-1]])
            assert_array_equal(seq._lengths, lengths)

        # Force ArraySequence constructor to use buffering.
        data = [rng.rand(rng.randint(10, 50), 3) for _ in range(10)]
        lengths = list(map(len, data))
        old_buffer_size = ArraySequence.BUFFER_SIZE
        ArraySequence.BUFFER_SIZE = 1
        seq = ArraySequence(data)
        assert_equal(len(seq), len(data))
        assert_equal(len(seq._offsets), len(data))
        assert_equal(len(seq._lengths), len(data))
        assert_equal(seq._data.shape[0], sum(lengths))
        assert_equal(seq._data.shape[1], 3)
        assert_array_equal(seq._offsets, np.r_[0, np.cumsum(lengths)[:-1]])
        assert_array_equal(seq._lengths, lengths)
        assert_equal(seq.common_shape, data[0].shape[1:])
        ArraySequence.BUFFER_SIZE = old_buffer_size

    def test_creating_arraysequence_from_generator(self):
        rng = np.random.RandomState(42)
        data = [rng.rand(rng.randint(10, 50), 3) for _ in range(10)]
        lengths = list(map(len, data))

        gen = (e for e in data)
        seq = ArraySequence(gen)
        assert_equal(len(seq), len(data))
        assert_equal(len(seq._offsets), len(data))
        assert_equal(len(seq._lengths), len(data))
        assert_equal(seq._data.shape[0], sum(lengths))
        assert_equal(seq._data.shape[1], 3)
        assert_array_equal(seq._offsets, np.r_[0, np.cumsum(lengths)[:-1]])
        assert_array_equal(seq._lengths, lengths)
        assert_equal(seq.common_shape, data[0].shape[1:])

        # Already consumed generator
        seq = ArraySequence(gen)
        assert_equal(len(seq), 0)
        assert_equal(len(seq._offsets), 0)
        assert_equal(len(seq._lengths), 0)
        assert_equal(seq._data.ndim, 0)
        assert_true(seq.common_shape == ())

    def test_creating_arraysequence_from_arraysequence(self):
        rng = np.random.RandomState(42)
        data = [rng.rand(rng.randint(10, 50), 3) for _ in range(10)]
        lengths = list(map(len, data))

        seq = ArraySequence(data)
        seq2 = ArraySequence(seq)
        assert_equal(len(seq2), len(data))
        assert_equal(len(seq2._offsets), len(data))
        assert_equal(len(seq2._lengths), len(data))
        assert_equal(seq2._data.shape[0], sum(lengths))
        assert_equal(seq2._data.shape[1], 3)
        assert_array_equal(seq2._offsets, np.r_[0, np.cumsum(lengths)[:-1]])
        assert_array_equal(seq2._lengths, lengths)
        assert_equal(seq2.common_shape, data[0].shape[1:])

    def test_arraysequence_iter(self):
        for e, d in zip(self.seq, self.data):
            assert_array_equal(e, d)

        # Try iterate through a corrupted ArraySequence object.
        seq = self.seq.copy()
        seq._lengths = seq._lengths[::2]
        assert_raises(ValueError, list, seq)

    def test_arraysequence_copy(self):
        seq = self.seq.copy()
        assert_array_equal(seq._data, self.seq._data)
        assert_true(seq._data is not self.seq._data)
        assert_array_equal(seq._offsets, self.seq._offsets)
        assert_true(seq._offsets is not self.seq._offsets)
        assert_array_equal(seq._lengths, self.seq._lengths)
        assert_true(seq._lengths is not self.seq._lengths)

        assert_equal(seq.common_shape, self.seq.common_shape)

        # When taking a copy of a `ArraySequence` generated by slicing.
        # Only needed data should be kept.
        seq = self.seq[::2].copy()

        assert_true(seq._data.shape[0] < self.seq._data.shape[0])
        assert_true(len(seq) < len(self.seq))
        assert_true(seq._data is not self.seq._data)
        assert_array_equal(seq._lengths, self.seq[::2]._lengths)
        assert_array_equal(seq._offsets,
                           np.cumsum(np.r_[0, self.seq[::2]._lengths])[:-1])
        assert_arrays_equal(seq, self.seq[::2])

    def test_arraysequence_append(self):
        # Maybe not necessary if `self.setUp` is always called before a
        # test method, anyways create a copy just in case.
        seq = self.seq.copy()

        rng = np.random.RandomState(1234)
        element = rng.rand(rng.randint(10, 50), *self.seq.common_shape)
        seq.append(element)
        assert_equal(len(seq), len(self.seq)+1)
        assert_equal(seq._offsets[-1], len(self.seq._data))
        assert_equal(seq._lengths[-1], len(element))
        assert_array_equal(seq._data[-len(element):], element)

        # Append with different shape.
        element = rng.rand(rng.randint(10, 50), 42)
        assert_raises(ValueError, seq.append, element)

        # Append to an empty ArraySequence.
        seq = ArraySequence()
        rng = np.random.RandomState(1234)
        shape = (2, 3, 4)
        element = rng.rand(rng.randint(10, 50), *shape)
        seq.append(element)

        assert_equal(len(seq), 1)
        assert_equal(seq._offsets[-1], 0)
        assert_equal(seq._lengths[-1], len(element))
        assert_array_equal(seq._data, element)
        assert_equal(seq.common_shape, shape)

    def test_arraysequence_extend(self):
        # Maybe not necessary if `self.setUp` is always called before a
        # test method, anyways create a copy just in case.
        seq = self.seq.copy()

        rng = np.random.RandomState(1234)
        shape = self.seq.common_shape
        new_data = [rng.rand(rng.randint(10, 50), *shape) for _ in range(10)]
        lengths = list(map(len, new_data))
        seq.extend(new_data)
        assert_equal(len(seq), len(self.seq)+len(new_data))
        assert_array_equal(seq._offsets[-len(new_data):],
                           len(self.seq._data) + np.cumsum([0] + lengths[:-1]))

        assert_array_equal(seq._lengths[-len(new_data):], lengths)
        assert_array_equal(seq._data[-sum(lengths):],
                           np.concatenate(new_data, axis=0))

        # Extend with another `ArraySequence` object.
        seq = self.seq.copy()
        new_seq = ArraySequence(new_data)
        seq.extend(new_seq)
        assert_equal(len(seq), len(self.seq)+len(new_seq))
        assert_array_equal(seq._offsets[-len(new_seq):],
                           len(self.seq._data) + np.cumsum(np.r_[0, lengths[:-1]]))

        assert_array_equal(seq._lengths[-len(new_seq):], lengths)
        assert_array_equal(seq._data[-sum(lengths):], new_seq._data)

        # Extend with another `ArraySequence` object that is a view (e.g. been sliced).
        # Need to make sure we extend only the data we need.
        seq = self.seq.copy()
        new_seq = ArraySequence(new_data)[::2]
        seq.extend(new_seq)
        assert_equal(len(seq), len(self.seq)+len(new_seq))
        assert_equal(len(seq._data), len(self.seq._data)+sum(new_seq._lengths))
        assert_array_equal(seq._offsets[-len(new_seq):],
                           len(self.seq._data) + np.cumsum(np.r_[0, new_seq._lengths[:-1]]))

        assert_array_equal(seq._lengths[-len(new_seq):], lengths[::2])
        assert_array_equal(seq._data[-sum(new_seq._lengths):], new_seq.copy()._data)
        assert_arrays_equal(seq[-len(new_seq):], new_seq)

        # Test extending an empty ArraySequence
        seq = ArraySequence()
        new_seq = ArraySequence(new_data)
        seq.extend(new_seq)
        assert_equal(len(seq), len(new_seq))
        assert_array_equal(seq._offsets, new_seq._offsets)
        assert_array_equal(seq._lengths, new_seq._lengths)
        assert_array_equal(seq._data, new_seq._data)

    def test_arraysequence_getitem(self):
        # Get one item
        for i, e in enumerate(self.seq):
            assert_array_equal(self.seq[i], e)

        # Get multiple items (this will create a view).
        indices = list(range(len(self.seq)))
        seq_view = self.seq[indices]
        assert_true(seq_view is not self.seq)
        assert_true(seq_view._data is self.seq._data)
        assert_true(seq_view._offsets is not self.seq._offsets)
        assert_true(seq_view._lengths is not self.seq._lengths)
        assert_array_equal(seq_view._offsets, self.seq._offsets)
        assert_array_equal(seq_view._lengths, self.seq._lengths)
        assert_arrays_equal(seq_view, self.seq)

        # Get multiple items using ndarray of data type.
        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            seq_view = self.seq[np.array(indices, dtype=dtype)]
            assert_true(seq_view is not self.seq)
            assert_true(seq_view._data is self.seq._data)
            assert_true(seq_view._offsets is not self.seq._offsets)
            assert_true(seq_view._lengths is not self.seq._lengths)
            assert_array_equal(seq_view._offsets, self.seq._offsets)
            assert_array_equal(seq_view._lengths, self.seq._lengths)
            for e1, e2 in zip_longest(seq_view, self.seq):
                assert_array_equal(e1, e2)

        # Get slice (this will create a view).
        seq_view = self.seq[::2]
        assert_true(seq_view is not self.seq)
        assert_true(seq_view._data is self.seq._data)
        assert_array_equal(seq_view._offsets, self.seq._offsets[::2])
        assert_array_equal(seq_view._lengths, self.seq._lengths[::2])
        for i, e in enumerate(seq_view):
            assert_array_equal(e, self.seq[i*2])

        # Use advance indexing with ndarray of data type bool.
        idx = np.array([False, True, True, False, True])
        seq_view = self.seq[idx]
        assert_true(seq_view is not self.seq)
        assert_true(seq_view._data is self.seq._data)
        assert_array_equal(seq_view._offsets,
                           self.seq._offsets[idx])
        assert_array_equal(seq_view._lengths,
                           self.seq._lengths[idx])
        assert_array_equal(seq_view[0], self.seq[1])
        assert_array_equal(seq_view[1], self.seq[2])
        assert_array_equal(seq_view[2], self.seq[4])

        # Test invalid indexing
        assert_raises(TypeError, self.seq.__getitem__, 'abc')

    def test_arraysequence_repr(self):
        # Test that calling repr on a ArraySequence object is not falling.
        repr(self.seq)

    def test_save_and_load_arraysequence(self):

        # Test saving and loading an empty ArraySequence.
        with tempfile.TemporaryFile(mode="w+b", suffix=".npz") as f:
            seq = ArraySequence()
            seq.save(f)
            f.seek(0, os.SEEK_SET)
            loaded_seq = ArraySequence.from_filename(f)
            assert_array_equal(loaded_seq._data, seq._data)
            assert_array_equal(loaded_seq._offsets, seq._offsets)
            assert_array_equal(loaded_seq._lengths, seq._lengths)

        # Test saving and loading a ArraySequence.
        with tempfile.TemporaryFile(mode="w+b", suffix=".npz") as f:
            rng = np.random.RandomState(42)
            data = [rng.rand(rng.randint(10, 50), 3) for _ in range(10)]
            seq = ArraySequence(data)
            seq.save(f)
            f.seek(0, os.SEEK_SET)
            loaded_seq = ArraySequence.from_filename(f)
            assert_array_equal(loaded_seq._data, seq._data)
            assert_array_equal(loaded_seq._offsets, seq._offsets)
            assert_array_equal(loaded_seq._lengths, seq._lengths)
