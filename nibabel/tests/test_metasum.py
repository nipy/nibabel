import random

import pytest
import numpy as np

from ..metasum import DimIndex, DimTypes, MetaSummary, ValueIndices


vidx_test_patterns = ([0] * 8,
                      ([0] * 4) + ([1] * 4),
                      [0, 0, 1, 2, 3, 3, 3, 4],
                      list(range(8)),
                      list(range(6)) + [6] * 2,
                      ([0] * 2) + list(range(2, 8)),
                      )


@pytest.mark.parametrize("in_list", vidx_test_patterns)
def test_value_indices_basics(in_list):
    '''Test basic ValueIndices behavior'''
    vidx = ValueIndices(in_list)
    assert vidx.n_input == len(in_list)
    assert len(vidx) == len(set(in_list))
    assert sorted(vidx.values()) == sorted(list(set(in_list)))
    for val in vidx.values():
        assert vidx.count(val) == in_list.count(val)
        for in_idx in vidx[val]:
            assert in_list[in_idx] == val == vidx.get_value(in_idx)
    out_list = vidx.to_list()
    assert in_list == out_list


@pytest.mark.parametrize("in_list", vidx_test_patterns)
def test_value_indices_append_extend(in_list):
    '''Test that append/extend are equivalent'''
    vidx_list = [ValueIndices() for _ in range(4)]
    vidx_list[0].extend(in_list)
    vidx_list[0].extend(in_list)
    for val in in_list:
        vidx_list[1].append(val)
    for val in in_list:
        vidx_list[1].append(val)
    vidx_list[2].extend(in_list)
    for val in in_list:
        vidx_list[2].append(val)
    for val in in_list:
        vidx_list[3].append(val)
    vidx_list[3].extend(in_list)
    for vidx in vidx_list:
        assert vidx.to_list() == in_list + in_list


metasum_test_dicts = (({'u1': 0, 'u2': 'a', 'u3': 3.0, 'c1': True, 'r1': 5},
                       {'u1': 2, 'u2': 'c', 'u3': 1.0, 'c1': True, 'r1': 5},
                       {'u1': 1, 'u2': 'b', 'u3': 2.0, 'c1': True, 'r1': 7},
                       ),
                      ({'u1': 0, 'u2': 'a', 'u3': 3.0, 'c1': True, 'r1': 5},
                       {'u1': 2, 'u2': 'c', 'c1': True, 'r1': 5},
                       {'u1': 1, 'u2': 'b', 'u3': 2.0, 'c1': True},
                       ),
                      )


@pytest.mark.parametrize("in_dicts", metasum_test_dicts)
def test_meta_summary_basics(in_dicts):
    msum = MetaSummary()
    all_keys = set()
    for in_dict in in_dicts:
        msum.append(in_dict)
        for key in in_dict.keys():
            all_keys.add(key)
    assert all_keys == set(msum.keys())
    for key in msum.const_keys():
        assert key.startswith('c')
    for key in msum.unique_keys():
        assert key.startswith('u')
    for key in msum.repeating_keys():
        assert key.startswith('r')
    for in_idx in range(len(in_dicts)):
        out_dict = msum.get_meta(in_idx)
        assert out_dict == in_dicts[in_idx]
        for key, in_val in in_dicts[in_idx].items():
            assert in_val == msum.get_val(in_idx, key)


def _make_nd_meta(shape, dim_info, const_meta=None):
    if const_meta is None:
        const_meta = {'series_number': '5'}
    meta_seq = []
    for nd_idx in np.ndindex(*shape):
        curr_meta = {}
        curr_meta.update(const_meta)
        for dim, dim_idx in zip(dim_info, nd_idx):
            curr_meta[dim.key] = dim_idx
        meta_seq.append(curr_meta)
    return meta_seq


ndsort_test_args = (((3,),
                     (DimIndex(DimTypes.SLICE, 'slice_location'),),
                     None),
                    ((3, 5),
                     (DimIndex(DimTypes.SLICE, 'slice_location'),
                      DimIndex(DimTypes.TIME, 'acq_time')),
                     None),
                    ((3, 5),
                     (DimIndex(DimTypes.SLICE, 'slice_location'),
                      DimIndex(DimTypes.PARAM, 'inversion_time')),
                     None),
                    ((3, 5, 7),
                     (DimIndex(DimTypes.SLICE, 'slice_location'),
                      DimIndex(DimTypes.TIME, 'acq_time'),
                      DimIndex(DimTypes.PARAM, 'echo_time')),
                     None),
                    ((3, 5, 7),
                     (DimIndex(DimTypes.SLICE, 'slice_location'),
                      DimIndex(DimTypes.PARAM, 'inversion_time'),
                      DimIndex(DimTypes.PARAM, 'echo_time')),
                     None),
                    ((5, 3),
                     (DimIndex(DimTypes.TIME, 'acq_time'),
                      DimIndex(DimTypes.PARAM, 'echo_time')),
                     None),
                    ((3, 5, 7),
                     (DimIndex(DimTypes.TIME, 'acq_time'),
                      DimIndex(DimTypes.PARAM, 'inversion_time'),
                      DimIndex(DimTypes.PARAM, 'echo_time')),
                     None),
                    ((5, 7),
                     (DimIndex(DimTypes.PARAM, 'inversion_time'),
                      DimIndex(DimTypes.PARAM, 'echo_time')),
                     None),
                    ((5, 7, 3),
                     (DimIndex(DimTypes.PARAM, 'inversion_time'),
                      DimIndex(DimTypes.PARAM, 'echo_time'),
                      DimIndex(DimTypes.PARAM, 'repetition_time')),
                     None),
                    )


@pytest.mark.parametrize("shape,dim_info,const_meta", ndsort_test_args)
def test_ndsort(shape, dim_info, const_meta):
    meta_seq = _make_nd_meta(shape, dim_info, const_meta)
    rand_idx_seq = [(i, m) for i, m in enumerate(meta_seq)]
    # TODO: Use some pytest plugin to manage randomness?  Just use fixed seed?
    random.shuffle(rand_idx_seq)
    rand_idx = [x[0] for x in rand_idx_seq]
    rand_seq = [x[1] for x in rand_idx_seq]
    msum = MetaSummary()
    for meta in rand_seq:
        msum.append(meta)
    out_shape, out_idxs = msum.nd_sort(dim_info)
    assert shape == out_shape
