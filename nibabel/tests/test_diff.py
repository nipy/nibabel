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

from ..tmpdirs import InTemporaryDirectory
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



DATA_PATH = abspath(pjoin(dirname(__file__), 'data'))

from cmdline.diff import diff_dicts


def test_diff_dicts():
    assert_equal(diff_dicts({}, {}), {})
    assert_equal(diff_dicts({'dtype': int}, {'dtype': float}), {'dtype': (int, float)})
    assert_equal(diff_dicts({1: 2}, {1: 2}), {})
    # TODO: mixed cases
    # TODO: arrays
