# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test scripts

Test running scripts
"""
from __future__ import division, print_function, absolute_import

from os.path import (dirname, join as pjoin, abspath)

from hypothesis import given
import hypothesis.strategies as st


DATA_PATH = abspath(pjoin(dirname(__file__), 'data'))

from nibabel.cmdline.diff import diff_values

# TODO: MAJOR TO DO IS TO FIGURE OUT HOW TO USE HYPOTHESIS FOR LONGER LIST LENGTHS WHILE STILL CONTROLLING FOR OUTCOMES

@given(st.data())
def test_diff_values_int(data):
    x = data.draw(st.integers(), label='x')
    y = data.draw(st.integers(min_value = x + 1), label='x+1')
    z = data.draw(st.integers(max_value = x - 1), label='x-1')
    list_1 = [x, x, x]
    list_2 = [x, y, x]
    list_3 = [x, y, z]
    list_4 = [x, z, x]
    list_5 = [y, z, y]
    list_6 = [y, x, y]

    assert diff_values('key', list_1) is None
    assert diff_values('key', list_2) == {'key': [x, y]}
    assert diff_values('key', list_3) == {'key': [x, y, z]}
    assert diff_values('key', list_4) == {'key': [x, z]}
    assert diff_values('key', list_5) == {'key': [y, z]}
    assert diff_values('key', list_6) == {'key': [y, x]}


@given(st.data())
def test_diff_values_float(data):
    x = data.draw(st.just(0), label='x')
    y = data.draw(st.floats(min_value = 1e8), label='y')
    z = data.draw(st.floats(max_value = -1e8), label='z')
    list_1 = [x, x, x]
    list_2 = [x, y, x]
    list_3 = [x, y, z]
    list_4 = [x, z, x]
    list_5 = [y, z, y]
    list_6 = [y, x, y]

    assert diff_values('key', list_1) is None
    assert diff_values('key', list_2) == {'key': [x, y]}
    assert diff_values('key', list_3) == {'key': [x, y, z]}
    assert diff_values('key', list_4) == {'key': [x, z]}
    assert diff_values('key', list_5) == {'key': [y, z]}
    assert diff_values('key', list_6) == {'key': [y, x]}


@given(st.data())
def test_diff_values_mixed(data):
    type_float = data.draw(st.floats(), label='float')
    type_int = data.draw(st.integers(), label='int')
    type_none = data.draw(st.none(), label='none')
    list_1 = [type_float, type_int]
    list_2 = [type_none, type_none, type_none]
    list_3 = [type_float, type_none, type_int]

    assert diff_values('key', list_1) == {'key': [type_float, type_int]}
    assert diff_values('key', list_2) is None
    assert diff_values('key', list_3) == {'key': [type_float, type_none, type_int]}


@given(st.data())
def test_diff_values_array(data):
    a = data.draw(st.lists(elements=st.integers(min_value=0), min_size=1))
    b = data.draw(st.lists(elements=st.integers(max_value=-1), min_size=1))
    c = data.draw(st.lists(elements=st.floats(min_value=1e8), min_size=1))
    d = data.draw(st.lists(elements=st.floats(max_value=-1e8), min_size=1))
    # TODO: Figure out a way to include 0 in lists (arrays)
    # TODO: Figure out a way to test for NaN

    list_1 = [a, a, b]
    list_2 = [a, b, c, d]
    list_3 = [c, d]
    list_4 = [b, b, c]

    assert diff_values('key', list_1) == {'key': [a, b]}
    assert diff_values('key', list_2) == {'key': [a, b, c, d]}
    assert diff_values('key', list_3) == {'key': [c, d]}
    assert diff_values('key', list_1) == {'key': [b, c]}