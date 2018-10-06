# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test diff
"""
from __future__ import division, print_function, absolute_import

from os.path import (dirname, join as pjoin, abspath)
import numpy as np


DATA_PATH = abspath(pjoin(dirname(__file__), 'data'))

from nibabel.cmdline.diff import are_values_different

# TODO: MAJOR TO DO IS TO FIGURE OUT HOW TO USE HYPOTHESIS FOR LONGER LIST LENGTHS WHILE STILL CONTROLLING FOR OUTCOMES


def test_diff_values_int():
    long = 10**30
    assert not are_values_different(0, 0)
    assert not are_values_different(1, 1)
    assert not are_values_different(long, long)
    assert are_values_different(0, 1)
    assert are_values_different(1, 2)
    assert are_values_different(1, long)


def test_diff_values_float():
    assert not are_values_different(0., 0.)
    assert not are_values_different(0., 0., 0.) # can take more
    assert not are_values_different(1.1, 1.1)
    assert are_values_different(0., 1.1)
    assert are_values_different(0., 0, 1.1)
    assert are_values_different(1., 2.)


def test_diff_values_mixed():
    assert are_values_different(1.0, 1)
    assert are_values_different(1.0, "1")
    assert are_values_different(1, "1")
    assert are_values_different(1, None)
    assert are_values_different(np.ndarray([0]), 'hey')
    assert not are_values_different(None, None)


def test_diff_values_array():
    from numpy import NaN, array, inf
    a_int = array([1, 2])
    a_float = a_int.astype(float)

    assert are_values_different(a_int, a_float)
    assert are_values_different(a_int, a_int, a_float)
    assert are_values_different(np.arange(3), np.arange(1, 4))
    assert are_values_different(np.arange(3), np.arange(4))
    assert are_values_different(np.arange(4), np.arange(4).reshape((2, 2)))
    # no broadcasting should kick in - shape difference
    assert are_values_different(array([1]), array([1, 1]))
    assert not are_values_different(a_int, a_int)
    assert not are_values_different(a_float, a_float)

    # NaNs - we consider them "the same" for the purpose of these comparisons
    assert not are_values_different(NaN, NaN)
    assert not are_values_different(NaN, NaN, NaN)
    assert are_values_different(NaN, NaN, 1)
    assert are_values_different(1, NaN, NaN)
    assert not are_values_different(array([NaN, NaN]), array([NaN, NaN]))
    assert not are_values_different(array([NaN, NaN]), array([NaN, NaN]), array([NaN, NaN]))
    assert not are_values_different(array([NaN, 1]), array([NaN, 1]))
    assert are_values_different(array([NaN, NaN]), array([NaN, 1]))
    assert are_values_different(array([0, NaN]), array([NaN, 0]))
    # and some inf should not be a problem
    assert not are_values_different(array([0, inf]), array([0, inf]))
    assert are_values_different(array([0, inf]), array([inf, 0]))
