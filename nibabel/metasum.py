# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Memory efficient tracking of meta data dicts with repeating elements
'''
from dataclasses import dataclass
from enum import IntEnum

from bitarray import bitarray, frozenbitarray
from bitarray.util import zeros


class FloatCanon:
    '''Look up a canonical float that we compare equal to'''

    def __init__(self, n_digits=6):
        self._n_digits = n_digits
        self._offset = 0.5 * (10 ** -n_digits)
        self._canon_vals = set()
        self._rounded = {}

    def get(self, val):
        '''Get a canonical value that at least compares equal to `val`'''
        res = self._values.get(val)
        if res is not None:
            return res
        lb = round(val, self._n_digits)
        res = self._rounded.get(lb)
        if res is not None:
            return res
        ub = round(val + self._offset, self._n_digits)
        res = self._rounded.get(ub)
        if res is not None:
            return res


_NoValue = object()

# TODO: Integrate some value canonicalization filtering? Or just require the
#       user to do that themselves?


class ValueIndices:
    """Track indices of values in sequence.

    If values repeat frequently then memory usage can be dramatically improved.
    It can be thought of as the inverse to a list.

    >>> values = ['a', 'a', 'b', 'a', 'b']
    >>> vidx = ValueIndices(values)
    >>> vidx['a']
    [0, 1, 3]
    >>> vidx['b']
    [2, 4]
    """

    def __init__(self, values=None):
        """Initialize a ValueIndices instance.

        Parameters
        ----------
        values : sequence
            The sequence of values to track indices on
        """

        self._n_input = 0

        # The values can be constant, unique to specific indices, or
        # arbitrarily varying
        self._const_val = _NoValue
        self._unique_vals = {}
        self._val_bitarrs = {}

        if values is not None:
            self.extend(values)

    @property
    def n_input(self):
        '''The number of inputs we are indexing'''
        return self._n_input

    def __len__(self):
        '''Number of unique values being tracked'''
        if self._const_val is not _NoValue:
            return 1
        return len(self._unique_vals) + len(self._val_bitarrs)

    def __getitem__(self, value):
        '''Return list of indices for the given value'''
        if self._const_val == value:
            return list(range(self._n_input))
        idx = self._unique_vals.get(value)
        if idx is not None:
            return [idx]
        ba = self._val_bitarrs[value]
        return list(self._extract_indices(ba))

    def values(self):
        '''Generate each unique value that has been seen'''
        if self._const_val is not _NoValue:
            yield self._const_val
            return
        for val in self._unique_vals.keys():
            yield val
        for val in self._val_bitarrs.keys():
            yield val

    def get_mask(self, value):
        '''Get bitarray mask of indices with this value'''
        if self._const_val is not _NoValue:
            if self._const_val != value:
                raise KeyError()
            res = bitarray(self._n_input)
            res.setall(1)
            return res
        idx = self._unique_vals.get(value)
        if idx is not None:
            res = zeros(self._n_inpuf)
            res[idx] = 1
            return res
        return self._val_bitarrs[value].copy()

    def count(self, value, mask=None):
        '''Number of indices for the given `value`'''
        if mask is not None:
            if len(mask) != self.n_input:
                raise ValueError("Mask length must match input length")
        if self._const_val is not _NoValue:
            if self._const_val != value:
                raise KeyError()
            if mask is None:
                return self._n_input
            return mask.count()
        unique_idx = self._unique_vals.get(value, _NoValue)
        if unique_idx is not _NoValue:
            if mask is not None:
                if mask[unique_idx]:
                    return 1
                return 0
            return 1
        if mask is not None:
            return (self._val_bitarrs[value] & mask).count()
        return self._val_bitarrs[value].count()

    def get_value(self, idx):
        '''Get the value at `idx`'''
        if not 0 <= idx < self._n_input:
            raise IndexError()
        if self._const_val is not _NoValue:
            return self._const_val
        for val, vidx in self._unique_vals.items():
            if vidx == idx:
                return val
        bit_idx = zeros(self._n_input)
        bit_idx[idx] = 1
        for val, ba in self._val_bitarrs.items():
            if (ba & bit_idx).any():
                return val
        assert False

    def to_list(self):
        '''Convert back to a list of values'''
        return [self.get_value(i) for i in range(self.n_input)]

    def extend(self, values):
        '''Add more values to the end of any existing ones'''
        init_size = self._n_input
        if isinstance(values, ValueIndices):
            other_is_vi = True
            other_size = values._n_input
        else:
            other_is_vi = False
            other_size = len(values)
        final_size = init_size + other_size
        for ba in self._val_bitarrs.values():
            ba.extend(zeros(other_size))
        if other_is_vi:
            if self._const_val is not _NoValue:
                if values._const_val is not _NoValue:
                    self._extend_const(values)
                    return
                else:
                    self._rm_const(final_size)
            elif values._const_val is not _NoValue:
                cval = values._const_val
                other_unique = {}
                other_bitarrs = {}
                if values._n_input == 1:
                    other_unique[cval] = 0
                else:
                    other_bitarrs[cval] = bitarray(values._n_input)
                    other_bitarrs[cval].setall(1)
            else:
                other_unique = values._unique_vals
                other_bitarrs = values._val_bitarrs
            for val, other_idx in other_unique.items():
                self._ingest_single(val, final_size, init_size, other_idx)
            for val, other_ba in other_bitarrs.items():
                curr_ba = self._val_bitarrs.get(val)
                if curr_ba is None:
                    curr_idx = self._unique_vals.get(val)
                    if curr_idx is None:
                        if init_size == 0:
                            new_ba = other_ba.copy()
                        else:
                            new_ba = zeros(init_size)
                            new_ba.extend(other_ba)
                    else:
                        new_ba = zeros(init_size)
                        new_ba[curr_idx] = True
                        new_ba.extend(other_ba)
                        del self._unique_vals[val]
                    self._val_bitarrs[val] = new_ba
                else:
                    curr_ba[init_size:] |= other_ba
                self._n_input += other_ba.count()
        else:
            for other_idx, val in enumerate(values):
                self._ingest_single(val, final_size, init_size, other_idx)
        assert self._n_input == final_size

    def append(self, value):
        '''Append another value as input'''
        if self._const_val == value:
            self._n_input += 1
            return
        elif self._const_val is not _NoValue:
            self._rm_const(self._n_input + 1)
            self._unique_vals[value] = self._n_input
            self._n_input += 1
            return
        if self._n_input == 0:
            self._const_val = value
            self._n_input += 1
            return
        curr_size = self._n_input
        found = False
        for val, bitarr in self._val_bitarrs.items():
            assert len(bitarr) == self._n_input
            if val == value:
                found = True
                bitarr.append(True)
            else:
                bitarr.append(False)
        if not found:
            curr_idx = self._unique_vals.get(value)
            if curr_idx is None:
                self._unique_vals[value] = curr_size
            else:
                new_ba = zeros(curr_size + 1)
                new_ba[curr_idx] = True
                new_ba[curr_size] = True
                self._val_bitarrs[value] = new_ba
                del self._unique_vals[value]
        self._n_input += 1

    def reverse(self):
        '''Reverse the indices in place'''
        for val, idx in self._unique_vals.items():
            self._unique_vals[val] = self._n_input - idx - 1
        for val, bitarr in self._val_bitarrs.items():
            bitarr.reverse()

    def argsort(self, reverse=False):
        '''Return array of indices in order that sorts the values'''
        if self._const_val is not _NoValue:
            return np.arange(self._n_input)
        res = np.empty(self._n_input, dtype=np.int64)
        vals = list(self._unique_vals.keys()) + list(self._val_bitarrs.keys())
        vals.sort(reverse=reverse)
        res_idx = 0
        for val in vals:
            idx = self._unique_vals.get(val)
            if idx is not None:
                res[res_idx] = idx
                res_idx += 1
                continue
            ba = self._val_bitarrs[val]
            for idx in self._extract_indices(ba):
                res[res_idx] = idx
                res_idx += 1
        return res

    def reorder(self, order):
        '''Reorder the indices in place'''
        if len(order) != self._n_input:
            raise ValueError("The 'order' has the incorrect length")
        for val, idx in self._unique_vals.items():
            self._unique_vals[val] = order.index(idx)
        for val, bitarr in self._val_bitarrs.items():
            new_ba = zeros(self._n_input)
            for idx in self._extract_indices(bitarr):
                new_ba[order.index(idx)] = True
            self._val_bitarrs[val] = new_ba

    def is_covariant(self, other):
        '''True if `other` has values that vary the same way ours do

        The actual values themselves are ignored
        '''
        if self._n_input != other._n_input or len(self) != len(other):
            return False
        if self._const_val is not _NoValue:
            return other._const_val is not _NoValue
        if self._n_input == len(self):
            return other._n_input == len(other)
        self_ba_set = set(frozenbitarray(ba) for ba in self._val_bitarrs.values())
        other_ba_set = set(frozenbitarray(ba) for ba in other._val_bitarrs.values())
        if self_ba_set != other_ba_set:
            return False
        if len(self._unique_vals) != len(other._unique_vals):
            return False
        return True

    def get_block_size(self):
        '''Return size of even blocks of values, or None if values aren't "blocked"

        The number of values must evenly divide the number of inputs into the block size,
        with each value appearing that same number of times.
        '''
        block_size, rem = divmod(self._n_input, len(self))
        if rem != 0:
            return None
        for val in self.values():
            if self.count(val) != block_size:
                return None
        return block_size

    def is_subpartition(self, other):
        ''''''

    def _extract_indices(self, ba):
        '''Generate integer indices from bitarray representation'''
        start = 0
        while True:
            try:
                # TODO: Is this the most efficient approach?
                curr_idx = ba.index(True, start)
            except ValueError:
                return
            yield curr_idx
            start = curr_idx + 1

    def _ingest_single(self, val, final_size, init_size, other_idx):
        '''Helper to ingest single value from another collection'''
        if val == self._const_val:
            self._n_input += 1
            return
        elif self._const_val is not _NoValue:
            self._rm_const(final_size)
        if self._n_input == 0:
            self._const_val = val
            self._n_input += 1
            return

        curr_ba = self._val_bitarrs.get(val)
        if curr_ba is None:
            curr_idx = self._unique_vals.get(val)
            if curr_idx is None:
                self._unique_vals[val] = init_size + other_idx
            else:
                new_ba = zeros(final_size)
                new_ba[curr_idx] = True
                new_ba[init_size + other_idx] = True
                self._val_bitarrs[val] = new_ba
                del self._unique_vals[val]
        else:
            curr_ba[init_size + other_idx] = True
        self._n_input += 1

    def _rm_const(self, final_size):
        assert self._const_val is not _NoValue
        if self._n_input == 1:
            self._unique_vals[self._const_val] = 0
        else:
            self._val_bitarrs[self._const_val] = zeros(final_size)
            self._val_bitarrs[self._const_val][:self._n_input] = True
        self._const_val = _NoValue

    def _extend_const(self, other):
        if self._const_val != other._const_val:
            if self._n_input == 1:
                self._unique_vals[self._const_val] = 0
            else:
                self_ba = bitarray(self._n_input)
                other_ba = bitarray(other._n_input)
                self_ba.setall(1)
                other_ba.setall(0)
                self._val_bitarrs[self._const_val] = self_ba + other_ba
            if other._n_input == 1:
                self._unique_vals[other._const_val] = self._n_input
            else:
                self_ba = bitarray(self._n_input)
                other_ba = bitarray(other._n_input)
                self_ba.setall(0)
                other_ba.setall(1)
                self._val_bitarrs[other._const_val] = self_ba + other_ba
            self._const_val = _NoValue
        self._n_input += other._n_input


_MissingKey = object()


class DimTypes(IntEnum):
    '''Enmerate the three types of nD dimensions'''
    SLICE = 1
    TIME = 2
    PARAM = 3


@dataclass
class DimIndex:
    '''Specify an nD index'''
    dim_type: DimTypes

    key: str


class NdSortError(Exception):
    '''Raised when the data cannot be sorted into an nD array as specified'''


class MetaSummary:
    '''Summarize a sequence of dicts, tracking how individual keys vary

    The assumption is that for any key many values will be constant, or at
    least repeated, and thus we can reduce memory consumption by only storing
    the value once along with the indices it appears at.
    '''

    def __init__(self):
        self._v_idxs = {}
        self._n_input = 0

    @property
    def n_input(self):
        return self._n_input

    def append(self, meta):
        seen = set()
        for key, v_idx in self._v_idxs.items():
            val = meta.get(key, _MissingKey)
            v_idx.append(val)
            seen.add(key)
        for key, val in meta.items():
            if key in seen:
                continue
            v_idx = ValueIndices([_MissingKey for _ in range(self._n_input)])
            v_idx.append(val)
            self._v_idxs[key] = v_idx
        self._n_input += 1

    def keys(self):
        '''Generate all known keys'''
        return self._v_idxs.keys()

    def const_keys(self):
        '''Generate keys with a constant value across all inputs'''
        for key, v_idx in self._v_idxs.items():
            if len(v_idx) == 1:
                yield key

    def unique_keys(self):
        '''Generate keys with a unique value in each input'''
        n_input = self._n_input
        if n_input <= 1:
            return
        for key, v_idx in self._v_idxs.items():
            if len(v_idx) == n_input:
                yield key

    def repeating_keys(self):
        '''Generate keys that have some repeating component but are not const
        '''
        n_input = self._n_input
        if n_input <= 1:
            return
        for key, v_idx in self._v_idxs.items():
            if 1 < len(v_idx) < n_input:
                yield key

    def covariant_groups(self, keys=None, block_only=False):
        '''Generate groups of keys that vary with the same pattern
        '''
        if keys is None:
            keys = self.keys()
        groups = []
        for key in keys:
            v_idx = self._v_idxs[key]
            if len(groups) == 0:
                groups.append((key, v_idx))
                continue
            for group in groups:
                if group[0][1].is_covariant(v_idx):
                    group.append(key)
                    break
            else:
                groups.append((key, v_idx))
        for group in groups:
            group[0] = group[0][0]
        return groups

    def get_meta(self, idx):
        '''Get the full dict at the given index'''
        res = {}
        for key, v_idx in self._v_idxs.items():
            val = v_idx.get_value(idx)
            if val is _MissingKey:
                continue
            res[key] = val
        return res

    def get_val(self, idx, key, default=None):
        '''Get the value at `idx` for the `key`, or return `default``'''
        res = self._v_idxs[key].get_value(idx)
        if res is _MissingKey:
            return default
        return res

    def reorder(self, order):
        '''Reorder indices in place'''
        for v_idx in self._v_idxs.values():
            v_idx.reorder(order)

    def nd_sort(self, dims):
        '''Produce linear indices to fill nD array as specified by `dims`

        Assumes each input corresponds to a 2D or 3D array, and the combined
        array is 3D+
        '''
        # Make sure dims aren't completely invalid
        if len(dims) == 0:
            raise ValueError("At least one dimension must be specified")
        last_dim = None
        for dim in dims:
            if last_dim is not None:
                if last_dim.dim_type > dim.dim_type:
                    # TODO: This only allows PARAM dimensions at the end, which I guess is reasonable?
                    raise ValueError("Invalid dimension order")
                elif last_dim.dim_type == dim.dim_type and dim.dim_type != DimTypes.PARAM:
                    raise ValueError("There can be at most one each of SLICE and TIME dimensions")
            last_dim = dim

        # Pull out info about different types of dims
        n_slices = None
        n_vol = None
        time_dim = None
        param_dims = []
        n_params = []
        total_params = 1
        shape = []
        curr_size = 1
        for dim in dims:
            dim_vidx = self._v_idxs[dim.key]
            dim_type = dim.dim_type
            if dim_type is DimTypes.SLICE:
                n_slices = len(dim_vidx)
                n_vol = dim_vidx.get_block_size()
                if n_vol is None:
                    raise NdSortError("There are missing or extra slices")
                shape.append(n_slices)
                curr_size *= n_slices
            elif dim_type is DimTypes.TIME:
                time_dim = dim
            elif dim_type is DimTypes.PARAM:
                if dim_vidx.get_block_size() is None:
                    raise NdSortError(f"The parameter {dim.key} doesn't evenly divide inputs")
                param_dims.append(dim)
                n_param = len(dim_vidx)
                n_params.append(n_param)
                total_params *= n_param
        if n_vol is None:
            n_vol = self._n_input

        # Size of the time dimension must be infered from the size of the other dims
        n_time = 1
        if time_dim is not None:
            n_time, rem = divmod(n_vol, total_params)
            if rem != 0:
                raise NdSortError(f"The combined parameters don't evenly divide inputs")
            shape.append(n_time)
            curr_size *= n_time

        # Complete the "shape", and do a more detailed check that our param dims make sense
        for dim, n_param in zip(param_dims, n_params):
            dim_vidx = self._v_idxs[dim.key]
            if dim_vidx.get_block_size() != curr_size:
                raise NdSortError(f"The parameter {dim.key} doesn't evenly divide inputs")
            shape.append(n_param)
            curr_size *= n_param

        # Extract dim keys for each input and do the actual sort
        sort_keys = [(idx, tuple(self.get_val(idx, dim.key) for dim in reversed(dims)))
                     for idx in range(self._n_input)]
        sort_keys.sort(key=lambda x: x[1])

        # TODO: Finish this
