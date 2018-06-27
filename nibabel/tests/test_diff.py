# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test diff
"""
from __future__ import division, print_function, absolute_import

from os.path import (dirname, join as pjoin, abspath)

from hypothesis import given
import hypothesis.strategies as st


DATA_PATH = abspath(pjoin(dirname(__file__), 'data'))

from nibabel.cmdline.diff import diff_values

# TODO: MAJOR TO DO IS TO FIGURE OUT HOW TO USE HYPOTHESIS FOR LONGER LIST
# LENGTHS WHILE STILL CONTROLLING FOR OUTCOMES


@given(st.data())
def test_diff_values_int(data):
    x = data.draw(st.integers(), label='x')
    y = data.draw(st.integers(min_value = x + 1), label='x+1')
    z = data.draw(st.integers(max_value = x - 1), label='x-1')

    assert not diff_values(x, x)
    assert diff_values(x, y)
    assert diff_values(x, z)
    assert diff_values(y, z)


@given(st.data())
def test_diff_values_float(data):
    x = data.draw(st.just(0), label='x')
    y = data.draw(st.floats(min_value=1e8), label='y')
    z = data.draw(st.floats(max_value=-1e8), label='z')

    assert not diff_values(x, x)
    assert diff_values(x, y)
    assert diff_values(x, z)
    assert diff_values(y, z)


@given(st.data())
def test_diff_values_mixed(data):
    type_float = data.draw(st.floats(), label='float')
    type_int = data.draw(st.integers(), label='int')
    type_none = data.draw(st.none(), label='none')

    assert diff_values(type_float, type_int)
    assert diff_values(type_float, type_none)
    assert diff_values(type_int, type_none)
    assert not diff_values(type_none, type_none)


@given(st.data())
def test_diff_values_array(data):
    a = data.draw(st.lists(elements=st.integers(min_value=0), min_size=1))
    b = data.draw(st.lists(elements=st.integers(max_value=-1), min_size=1))
    c = data.draw(st.lists(elements=st.floats(min_value=1e8), min_size=1))
    d = data.draw(st.lists(elements=st.floats(max_value=-1e8), min_size=1))
    # TODO: Figure out a way to include 0 in lists (arrays)

    assert diff_values(a, b)
    assert diff_values(c, d)
    assert not diff_values(a, a)
