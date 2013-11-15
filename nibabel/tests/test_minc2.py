# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*- # vi: set ft=python sts=4 ts=4 sw=4 et: ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ## #
#   See COPYING file distributed along with the NiBabel package for the #   copyright and license terms. #
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ## from __future__ import division, print_function, absolute_import

import os

import numpy as np

from ..optpkg import optional_package

h5py, have_h5py, setup_module = optional_package('h5py')

from .. import minc2
from ..minc2 import Minc2File, Minc2Image

from nose.tools import (assert_true, assert_equal, assert_false, assert_raises)
from numpy.testing import assert_array_equal, assert_array_almost_equal

from ..testing import data_path

from . import test_minc1 as tm2


if have_h5py:
    class TestMinc2File(tm2._TestMincFile):
        module = minc2
        file_class = Minc2File
        opener = h5py.File
        example_params = dict(
            fname = os.path.join(data_path, 'small.mnc'),
            shape = (18, 28, 29),
            type = np.int16,
            affine = np.array([[0, 0, 7.0, -98],
                            [0, 8.0, 0, -134],
                            [9.0, 0, 0, -72],
                            [0, 0, 0, 1]]),
            zooms = (9., 8., 7.),
            # These values from mincstats
            min = 0.1185331417,
            max = 92.87690699,
            mean = 31.2127952)


    class TestMinc2Image(tm2.TestMinc1Image):
        image_class = Minc2Image
