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

from ..arrayproxy import ArrayProxy, is_proxy
from ..nifti1 import Nifti1Header

from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_not_equal, assert_raises)
from nibabel.testing import VIRAL_MEMMAP

from .test_fileslice import slicer_samples


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


class CArrayProxy(ArrayProxy):
    # C array memory layout
    order = 'C'


def test_init():
    bio = BytesIO()
    shape = [2, 3, 4]
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
    # C order also possible
    bio = BytesIO()
    bio.seek(16)
    bio.write(arr.tostring(order='C'))
    ap = CArrayProxy(bio, FunkyHeader((2, 3, 4)))
    assert_array_equal(np.asarray(ap), arr)


def write_raw_data(arr, hdr, fileobj):
    hdr.set_data_shape(arr.shape)
    hdr.set_data_dtype(arr.dtype)
    fileobj.write(b'\x00' * hdr.get_data_offset())
    fileobj.write(arr.tostring(order='F'))


def test_nifti1_init():
    bio = BytesIO()
    shape = (2, 3, 4)
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


def test_proxy_slicing():
    shapes = (15, 16, 17)
    for n_dim in range(1, len(shapes) + 1):
        shape = shapes[:n_dim]
        arr = np.arange(np.prod(shape)).reshape(shape)
        for offset in (0, 20):
            hdr = Nifti1Header()
            hdr.set_data_offset(offset)
            hdr.set_data_dtype(arr.dtype)
            hdr.set_data_shape(shape)
            for order, klass in ('F', ArrayProxy), ('C', CArrayProxy):
                fobj = BytesIO()
                fobj.write(b'\0' * offset)
                fobj.write(arr.tostring(order=order))
                prox = klass(fobj, hdr)
                for sliceobj in slicer_samples(shape):
                    assert_array_equal(arr[sliceobj], prox[sliceobj])
    # Check slicing works with scaling
    hdr.set_slope_inter(2.0, 1.0)
    fobj = BytesIO()
    fobj.write(b'\0' * offset)
    fobj.write(arr.tostring(order='F'))
    prox = ArrayProxy(fobj, hdr)
    sliceobj = (None, slice(None), 1, -1)
    assert_array_equal(arr[sliceobj] * 2.0 + 1.0, prox[sliceobj])


def test_is_proxy():
    # Test is_proxy function
    hdr = FunkyHeader((2, 3, 4))
    bio = BytesIO()
    prox = ArrayProxy(bio, hdr)
    assert_true(is_proxy(prox))
    assert_false(is_proxy(bio))
    assert_false(is_proxy(hdr))
    assert_false(is_proxy(np.zeros((2, 3, 4))))

    class NP(object):
        is_proxy = False
    assert_false(is_proxy(NP()))


def test_get_unscaled():
    # Test fetch of raw array
    class FunkyHeader2(FunkyHeader):

        def get_slope_inter(self):
            return 2.1, 3.14
    shape = (2, 3, 4)
    hdr = FunkyHeader2(shape)
    bio = BytesIO()
    # Check standard read works
    arr = np.arange(24, dtype=np.int32).reshape(shape, order='F')
    bio.write(b'\x00' * hdr.get_data_offset())
    bio.write(arr.tostring(order='F'))
    prox = ArrayProxy(bio, hdr)
    assert_array_almost_equal(np.array(prox), arr * 2.1 + 3.14)
    # Check unscaled read works
    assert_array_almost_equal(prox.get_unscaled(), arr)


def test_mmap():
    # Unscaled should return mmap from suitable file, this can be tuned
    hdr = FunkyHeader((2, 3, 4))
    check_mmap(hdr, hdr.get_data_offset(), ArrayProxy)


def check_mmap(hdr, offset, proxy_class,
               has_scaling=False,
               unscaled_is_view=True):
    """ Assert that array proxies return memory maps as expected

    Parameters
    ----------
    hdr : object
        Image header instance
    offset : int
        Offset in bytes of image data in file (that we will write)
    proxy_class : class
        Class of image array proxy to test
    has_scaling : {False, True}
        True if the `hdr` says to apply scaling to the output data, False
        otherwise.
    unscaled_is_view : {True, False}
        True if getting the unscaled data returns a view of the array.  If
        False, then type of returned array will depend on whether numpy has the
        old viral (< 1.12) memmap behavior (returns memmap) or the new behavior
        (returns ndarray).  See: https://github.com/numpy/numpy/pull/7406
    """
    shape = hdr.get_data_shape()
    arr = np.arange(np.prod(shape), dtype=hdr.get_data_dtype()).reshape(shape)
    fname = 'test.bin'
    # Whether unscaled array memory backed by memory map (regardless of what
    # numpy says).
    unscaled_really_mmap = unscaled_is_view
    # Whether scaled array memory backed by memory map (regardless of what
    # numpy says).
    scaled_really_mmap = unscaled_really_mmap and not has_scaling
    with InTemporaryDirectory():
        with open(fname, 'wb') as fobj:
            fobj.write(b' ' * offset)
            fobj.write(arr.tostring(order='F'))
        for mmap, expected_mode in (
                # mmap value, expected memmap mode
                # mmap=None -> no mmap value
                # expected mode=None -> no memmap returned
                (None, 'c'),
                (True, 'c'),
                ('c', 'c'),
                ('r', 'r'),
                (False, None)):
            kwargs = {}
            if mmap is not None:
                kwargs['mmap'] = mmap
            prox = proxy_class(fname, hdr, **kwargs)
            unscaled = prox.get_unscaled()
            back_data = np.asanyarray(prox)
            unscaled_is_mmap = isinstance(unscaled, np.memmap)
            back_is_mmap =  isinstance(back_data, np.memmap)
            if expected_mode is None:
                assert_false(unscaled_is_mmap)
                assert_false(back_is_mmap)
            else:
                assert_equal(unscaled_is_mmap,
                             VIRAL_MEMMAP or unscaled_really_mmap)
                assert_equal(back_is_mmap,
                             VIRAL_MEMMAP or scaled_really_mmap)
                if scaled_really_mmap:
                    assert_equal(back_data.mode, expected_mode)
            del prox, back_data
            # Check that mmap is keyword-only
            assert_raises(TypeError, proxy_class, fname, hdr, True)
            # Check invalid values raise error
            assert_raises(ValueError, proxy_class, fname, hdr, mmap='rw')
            assert_raises(ValueError, proxy_class, fname, hdr, mmap='r+')
