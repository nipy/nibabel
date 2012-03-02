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
from __future__ import with_statement

from copy import deepcopy

from ..py3k import BytesIO, ZEROB, asbytes
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

    def copy(self):
        return self.__class__(self.shape[:])

    def get_data_shape(self):
        return self.shape[:]

    def data_from_fileobj(self, fileobj):
        return np.arange(np.prod(self.shape)).reshape(self.shape)


def test_init():
    bio = BytesIO()
    shape = [2,3,4]
    hdr = FunkyHeader(shape)
    ap = ArrayProxy(bio, hdr)
    assert_true(ap.file_like is bio)
    assert_equal(ap.shape, shape)
    # shape should be read only
    assert_raises(AttributeError, setattr, ap, 'shape', shape)
    # Check there has been a copy of the header
    assert_false(ap.header is hdr)
    # Check we can modify the original header without changing the ap version
    hdr.shape[0] = 6
    assert_not_equal(ap.shape, shape)
    # Get the data
    assert_array_equal(np.asarray(ap), np.arange(24).reshape((2,3,4)))


def write_raw_data(arr, hdr, fileobj):
    hdr.set_data_shape(arr.shape)
    hdr.set_data_dtype(arr.dtype)
    fileobj.write(ZEROB * hdr.get_data_offset())
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
