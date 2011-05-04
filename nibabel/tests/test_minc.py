# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
from __future__ import with_statement

import os
import gzip
import bz2

import numpy as np

from ..externals.netcdf import netcdf_file as netcdf

from .. import load, Nifti1Image
from ..minc import MincFile

from nose.tools import (assert_true, assert_equal, assert_false, assert_raises)
from numpy.testing import assert_array_equal, assert_array_almost_equal
from ..tmpdirs import InTemporaryDirectory
from ..testing import data_path

MINC_EXAMPLE = dict(
    fname = os.path.join(data_path, 'tiny.mnc'),
    shape = (10,20,20),
    affine = np.array([[0, 0, 2.0, -20],
                       [0, 2.0, 0, -20],
                       [2.0, 0, 0, -10],
                       [0, 0, 0, 1]]),
    # These values from SPM2
    min = 0.20784314,
    max = 0.74901961,
    mean = 0.60602819)


def test_mincfile():
    mnc = MincFile(netcdf(MINC_EXAMPLE['fname'], 'r'))
    assert_equal(mnc.get_data_dtype().type, np.uint8)
    assert_equal(mnc.get_data_shape(), MINC_EXAMPLE['shape'])
    assert_equal(mnc.get_zooms(), (2.0, 2.0, 2.0))
    assert_array_equal(mnc.get_affine(), MINC_EXAMPLE['affine'])
    data = mnc.get_scaled_data()
    assert_equal(data.shape, MINC_EXAMPLE['shape'])


def test_load():
    # Check highest level load of minc works
    img = load(MINC_EXAMPLE['fname'])
    data = img.get_data()
    assert_equal(data.shape, MINC_EXAMPLE['shape'])
    # min, max, mean values from read in SPM2
    assert_array_almost_equal(data.min(), 0.20784314)
    assert_array_almost_equal(data.max(), 0.74901961)
    assert_array_almost_equal(data.mean(), 0.60602819)
    # check if mnc can be converted to nifti
    ni_img = Nifti1Image.from_image(img)
    assert_array_equal(ni_img.get_affine(), MINC_EXAMPLE['affine'])
    assert_array_equal(ni_img.get_data(), data)


def test_compressed():
    # we can read minc compressed
    content = open(MINC_EXAMPLE['fname'], 'rb').read()
    openers_exts = ((gzip.open, '.gz'), (bz2.BZ2File, '.bz2'))
    with InTemporaryDirectory():
        for opener, ext in openers_exts:
            fname = 'test.mnc' + ext
            fobj = opener(fname, 'wb')
            fobj.write(content)
            fobj.close()
            img = load(fname)
            data = img.get_data()
            assert_array_almost_equal(data.mean(), 0.60602819)
            del img
