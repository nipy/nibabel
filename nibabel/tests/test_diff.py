# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test scripts

Test running scripts
"""
from __future__ import division, print_function, absolute_import

import sys
import os
from os.path import (dirname, join as pjoin, abspath, splitext, basename,
                     exists)
import csv
from glob import glob

import numpy as np
from numpy import array

# from ..tmpdirs import InTemporaryDirectory
from ..loadsave import load
from ..orientations import flip_axis, aff2axcodes, inv_ornt_aff

from nose.tools import assert_true, assert_false, assert_equal
from nose import SkipTest

from numpy.testing import assert_almost_equal

from .scriptrunner import ScriptRunner
from .nibabel_data import needs_nibabel_data
from ..testing import assert_dt_equal, assert_re_in
from .test_parrec import (DTI_PAR_BVECS, DTI_PAR_BVALS,
                          EXAMPLE_IMAGES as PARREC_EXAMPLES)
from .test_parrec_data import BALLS, AFF_OFF
from .test_helpers import assert_data_similar

from hypothesis import given
import hypothesis.strategies as st


DATA_PATH = abspath(pjoin(dirname(__file__), 'data'))

from nibabel.cmdline.diff import diff_dicts


@given(st.data())
def test_diff_dicts_int(data):
    x = data.draw(st.integers(), label='x')
    y = data.draw(st.integers(min_value = x + 1), label='x+1')
    z = data.draw(st.integers(max_value = x - 1), label='x-1')

    assert diff_dicts('key', x, x) is None
    assert diff_dicts('key', x, y) == {'key': (x, y)}
    assert diff_dicts('key', x, z) == {'key': (x, z)}
    assert diff_dicts('key', y, z) == {'key': (y, z)}


@given(st.data())
def test_diff_dicts_float(data):
    x = data.draw(st.just(0), label='x')
    y = data.draw(st.floats(min_value = 1e8), label='y')
    z = data.draw(st.floats(max_value = -1e8), label='z')

    assert diff_dicts('key', x, x) is None
    assert diff_dicts('key', x, y) == {'key': (x, y)}
    assert diff_dicts('key', x, z) == {'key': (x, z)}
    assert diff_dicts('key', y, z) == {'key': (y, z)}


@given(st.data())
def test_diff_dicts_mixed(data):
    type_float = data.draw(st.floats(), label='float')
    type_int = data.draw(st.integers(), label='int')
    type_none = data.draw(st.none(), label='none')

    assert diff_dicts('key', type_float, type_int) == {'key': (type_float, type_int)}
    assert diff_dicts('key', type_float, type_none) == {'key': (type_float, type_none)}
    assert diff_dicts('key', type_int, type_none) == {'key': (type_int, type_none)}
    assert diff_dicts('key', type_none, type_none) is None


@given(st.data())
def test_diff_dicts_array(data):
    a = data.draw(st.lists(elements=st.integers(min_value=0), min_size=1))
    b = data.draw(st.lists(elements=st.integers(max_value=-1), min_size=1))
    c = data.draw(st.lists(elements=st.floats(min_value=1e8), min_size=1))
    d = data.draw(st.lists(elements=st.floats(max_value=-1e8), min_size=1))
    # TODO: Figure out a way to include 0 in lists (arrays)

    assert diff_dicts('key', a, b) == {'key': (a, b)}
    assert diff_dicts('key', c, d) == {'key': (c, d)}