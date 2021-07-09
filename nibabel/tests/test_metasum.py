from ..metasum import MetaSummary, ValueIndices

import pytest


vidx_test_patterns = ([0] * 8,
                      ([0] * 4) + ([1] * 4),
                      [0, 0, 1, 2, 3, 3, 3, 4],
                      list(range(8)),
                      list(range(6)) + [6] * 2,
                      ([0] * 2) + list(range(2, 8)),
                      )


@pytest.mark.parametrize("in_list", vidx_test_patterns)
def test_value_indices_basics(in_list):
    '''Test we can roundtrip list -> ValueIndices -> list'''
    vidx = ValueIndices(in_list)
    assert vidx.n_input == len(in_list)
    assert len(vidx) == len(set(in_list))
    assert sorted(vidx.values()) == sorted(list(set(in_list)))
    for val in vidx.values():
        assert vidx.count(val) == in_list.count(val)
        for in_idx in vidx[val]:
            assert in_list[in_idx] == val
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
