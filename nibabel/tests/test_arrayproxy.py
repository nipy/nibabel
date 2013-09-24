# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Tests for arrayproxy module
"""
from __future__ import division, print_function, absolute_import

import warnings

from ..externals.six import BytesIO
from ..tmpdirs import InTemporaryDirectory

import numpy as np

from ..arrayproxy import ArrayProxy
from ..nifti1 import Nifti1Header

from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_not_equal, assert_raises)


class FunkyHeader(object):
    def __init__(self, shape):
        self.shape = shape

    def get_data_shape(self):
        return self.shape[:]

    def get_data_dtype(self):
        return np.int32

    def get_data_offset(self):
        return 16

    def get_slope_inter(self):
        return 1.0, 0.0

    def copy(self):
        # Not needed when we remove header property
        return FunkyHeader(self.shape)


def test_init():
    bio = BytesIO()
    shape = [2,3,4]
    dtype = np.int32
    arr = np.arange(24, dtype=dtype).reshape(shape)
    bio.seek(16)
    bio.write(arr.tostring(order='F'))
    hdr = FunkyHeader(shape)
    ap = ArrayProxy(bio, hdr)
    assert_true(ap.file_like is bio)
    assert_equal(ap.shape, shape)
    # shape should be read only
    assert_raises(AttributeError, setattr, ap, 'shape', shape)
    # Get the data
    assert_array_equal(np.asarray(ap), arr)
    # Check we can modify the original header without changing the ap version
    hdr.shape[0] = 6
    assert_not_equal(ap.shape, shape)
    # Data stays the same, also
    assert_array_equal(np.asarray(ap), arr)


def write_raw_data(arr, hdr, fileobj):
    hdr.set_data_shape(arr.shape)
    hdr.set_data_dtype(arr.dtype)
    fileobj.write(b'\x00' * hdr.get_data_offset())
    fileobj.write(arr.tostring(order='F'))


def test_nifti1_init():
    bio = BytesIO()
    shape = (2,3,4)
    hdr = Nifti1Header()
    arr = np.arange(24, dtype=np.int16).reshape(shape)
    write_raw_data(arr, hdr, bio)
    hdr.set_slope_inter(2, 10)
    ap = ArrayProxy(bio, hdr)
    assert_true(ap.file_like == bio)
    assert_equal(ap.shape, shape)
    # Check there has been a copy of the header
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_false(ap.header is hdr)
    # Get the data
    assert_array_equal(np.asarray(ap), arr * 2.0 + 10)
    with InTemporaryDirectory():
        f = open('test.nii', 'wb')
        write_raw_data(arr, hdr, f)
        f.close()
        ap = ArrayProxy('test.nii', hdr)
        assert_true(ap.file_like == 'test.nii')
        assert_equal(ap.shape, shape)
        assert_array_equal(np.asarray(ap), arr * 2.0 + 10)
