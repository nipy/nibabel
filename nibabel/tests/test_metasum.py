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
def test_value_indices_rt(in_list):
    '''Test we can roundtrip list -> ValueIndices -> list'''
    vidx = ValueIndices(in_list)
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


metasum_test_dicts = (({'key1': 0, 'key2': 'a', 'key3': 3.0},
                       {'key1': 2, 'key2': 'c', 'key3': 1.0},
                       {'key1': 1, 'key2': 'b', 'key3': 2.0},
                       ),
                      ({'key1': 0, 'key2': 'a', 'key3': 3.0},
                       {'key1': 2, 'key2': 'c'},
                       {'key1': 1, 'key2': 'b', 'key3': 2.0},
                       ),
                      )


@pytest.mark.parametrize("in_dicts", metasum_test_dicts)
def test_meta_summary_rt(in_dicts):
    msum = MetaSummary()
    for in_dict in in_dicts:
        msum.append(in_dict)
    for in_idx in range(len(in_dicts)):
        out_dict = msum.get_meta(in_idx)
        assert out_dict == in_dicts[in_idx]
        for key, in_val in in_dicts[in_idx].items():
            assert in_val == msum.get_val(in_idx, key)
