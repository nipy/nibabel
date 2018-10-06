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
    a_int = np.array([1, 2])
    a_float = a_int.astype(float)

    #assert are_values_different(a_int, a_float)
    assert are_values_different(np.arange(3), np.arange(1, 4))
    assert not are_values_different(a_int, a_int)
    assert not are_values_different(a_float, a_float)
