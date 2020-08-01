# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Aggregate information for mutliple images
'''
from bitarray import bitarray, frozenbitarray
from bitarry.utils import zeroes


class FloatCanon(object):
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
class ValueIndices(object):
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
            res = zeroes(self._n_inpuf)
            res[idx] = 1
            return res
        return self._val_bitarrs[value].copy()

    def num_indices(self, value):
        '''Number of indices for the given `value`'''
        if self._const_val is not _NoValue:
            if self._const_val != value:
                raise KeyError()
            return self._n_input
        if value in self._unique_vals:
            return 1
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
        bit_idx = zeroes(self._n_input)
        bit_idx[idx] = 1
        for val, ba in self._val_bitarrs.items():
            if (ba | bit_idx).any():
                return val
        assert False

    def extend(self, values):
        '''Add more values to the end of any existing ones'''
        curr_size = self._n_input
        if isinstance(values, ValueIndices):
            other_is_vi = True
            other_size = values._n_input
        else:
            other_is_vi = False
            other_size = len(values)
        final_size = curr_size + other_size
        for ba in self._val_bitarrs.values():
            ba.extend(zeroes(other_size))
        if other_is_vi:
            if self._const_val is not _NoValue:
                if values._const_val is not _NoValue:
                    self._extend_const(values)
                    return
                else:
                    self._rm_const()
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
                self._ingest_single(val, final_size, curr_size, other_idx)
            for val, other_ba in other_bitarrs.items():
                curr_ba = self._val_bitarrs.get(val)
                if curr_ba is None:
                    curr_idx = self._unique_vals.get(val)
                    if curr_idx is None:
                        if curr_size == 0:
                            new_ba = other_ba.copy()
                        else:
                            new_ba = zeroes(curr_size)
                            new_ba.extend(other_ba)
                    else:
                        new_ba = zeroes(curr_size)
                        new_ba[curr_idx] = True
                        new_ba.extend(other_ba)
                        del self._unique_vals[val]
                    self._val_bitarrs[val] = new_ba
                else:
                    curr_ba[curr_size:] |= other_ba
        else:
            for other_idx, val in enumerate(values):
                self._ingest_single(val, final_size, curr_size, other_idx)
        self._n_input = final_size

    def append(self, value):
        '''Append another value as input'''
        if self._const_val == value:
            self._n_input += 1
            return
        elif self._const_val is not _NoValue:
            self._rm_const()
        curr_size = self._n_input
        found = False
        for val, bitarr in self._val_bitarrs.items():
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
                new_ba = zeroes(curr_size + 1)
                new_ba[curr_idx] = True
                new_ba[curr_size] = True
                self._val_bitarrs[value] = new_ba
                del self._unique_vals[value]
        self._n_input += 1

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

    def is_blocked(self, block_factor=None):
        '''True if each value has the same number of indices

        If `block_factor` is not None we also test that it evenly divides the
        block size.
        '''
        block_size, rem = divmod(self._n_input, len(self))
        if rem != 0:
            return False
        if block_factor is not None and block_size % block_factor != 0:
            return False
        for val in self.values():
            if self.num_indices(val) != block_size:
                return False
        return True

    def is_subpartition(self, other):
        '''True if we have more values and they nest within values from other


        '''

    def _extract_indices(self, ba):
        '''Generate integer indices from bitarray representation'''
        start = 0
        while True:
            try:
                # TODO: Is this the most efficient approach?
                curr_idx = ba.index(True, start=start)
            except ValueError:
                return
            yield curr_idx
            start = curr_idx

    def _ingest_single(self, val, final_size, curr_size, other_idx):
        '''Helper to ingest single value from another collection'''
        curr_ba = self._val_bitarrs.get(val)
        if curr_ba is None:
            curr_idx = self._unique_vals.get(val)
            if curr_idx is None:
                self._unique_vals[val] = curr_size + other_idx
            else:
                new_ba = zeroes(final_size)
                new_ba[curr_idx] = True
                new_ba[curr_size + other_idx] = True
                self._val_bitarrs = new_ba
                del self._unique_vals[val]
        else:
            curr_ba[curr_size + other_idx] = True

    def _rm_const(self):
        assert self._const_val is not _NoValue
        if self._n_input == 1:
            self._unique_vals[self._const_val] = 0
        else:
            self._val_bitarrs[self._const_val] = bitarray(self._n_input)
            self._val_bitarrs[self._const_val].setall(1)
        self._const_val == _NoValue

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

    def extend(self, metas):
        pass # TODO

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

    def repeating_groups(self, block_only=False, block_factor=None):
        '''Generate groups of repeating keys that vary with the same pattern
        '''
        n_input = self._n_input
        if n_input <= 1:
            # If there is only one element, consider all keys as const
            return
        # TODO: Can we sort so grouped v_idxs are sequential?
        #         - Sort by num values isn't sufficient
        curr_group = []
        for key, v_idx in self._v_idxs.items():
            if 1 < len(v_idx) < n_input:
                if v_idx.is_even(block_factor):
                pass # TODO

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
        res = self._v_idxs[key].get_value(key)
        if res is _MissingKey:
            return default
        return res

    def nd_sort(self, dim_keys=None):
        '''Produce indices ordered so as to fill an n-D array'''

class SummaryTree:
    '''Groups incoming meta data and creates hierarchy of related groups

    Each leaf node in the tree is a `MetaSummary`
    '''
    def __init__(self, group_keys):
        self._group_keys = group_keys
        self._group_summaries= {}

    def add(self, meta):
        pass

    def groups(self):
        '''Generate the groups and their meta summaries'''

