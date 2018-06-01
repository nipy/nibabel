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
import gzip
import contextlib

import pickle
from io import BytesIO
from ..tmpdirs import InTemporaryDirectory

import numpy as np

from ..arrayproxy import (ArrayProxy, is_proxy, reshape_dataobj)
from ..openers import ImageOpener
from ..nifti1 import Nifti1Header

import mock

from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_not_equal, assert_raises)
from nibabel.testing import memmap_after_ufunc

from .test_fileslice import slicer_samples
from .test_openers import patch_indexed_gzip


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
    # Illegal init
    assert_raises(TypeError, ArrayProxy, bio, object())


def test_tuplespec():
    bio = BytesIO()
    shape = [2, 3, 4]
    dtype = np.int32
    arr = np.arange(24, dtype=dtype).reshape(shape)
    bio.seek(16)
    bio.write(arr.tostring(order='F'))
    # Create equivalent header and tuple specs
    hdr = FunkyHeader(shape)
    tuple_spec = (hdr.get_data_shape(), hdr.get_data_dtype(),
                  hdr.get_data_offset(), 1., 0.)
    ap_header = ArrayProxy(bio, hdr)
    ap_tuple = ArrayProxy(bio, tuple_spec)
    # Header and tuple specs produce identical behavior
    for prop in ('shape', 'dtype', 'offset', 'slope', 'inter', 'is_proxy'):
        assert_equal(getattr(ap_header, prop), getattr(ap_tuple, prop))
    for method, args in (('get_unscaled', ()), ('__array__', ()),
                         ('__getitem__', ((0, 2, 1), ))
                         ):
        assert_array_equal(getattr(ap_header, method)(*args),
                           getattr(ap_tuple, method)(*args))
    # Tuple-defined ArrayProxies have no header to store
    with warnings.catch_warnings():
        assert_true(ap_tuple.header is None)
    # Partial tuples of length 2-4 are also valid
    for n in range(2, 5):
        ArrayProxy(bio, tuple_spec[:n])
    # Bad tuple lengths
    assert_raises(TypeError, ArrayProxy, bio, ())
    assert_raises(TypeError, ArrayProxy, bio, tuple_spec[:1])
    assert_raises(TypeError, ArrayProxy, bio, tuple_spec + ('error',))


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


def test_reshape_dataobj():
    # Test function that reshapes using method if possible
    shape = (1, 2, 3, 4)
    hdr = FunkyHeader(shape)
    bio = BytesIO()
    prox = ArrayProxy(bio, hdr)
    arr = np.arange(np.prod(shape), dtype=prox.dtype).reshape(shape)
    bio.write(b'\x00' * prox.offset + arr.tostring(order='F'))
    assert_array_equal(prox, arr)
    assert_array_equal(reshape_dataobj(prox, (2, 3, 4)),
                       np.reshape(arr, (2, 3, 4)))
    assert_equal(prox.shape, shape)
    assert_equal(arr.shape, shape)
    assert_array_equal(reshape_dataobj(arr, (2, 3, 4)),
                       np.reshape(arr, (2, 3, 4)))
    assert_equal(arr.shape, shape)

    class ArrGiver(object):

        def __array__(self):
            return arr

    assert_array_equal(reshape_dataobj(ArrGiver(), (2, 3, 4)),
                       np.reshape(arr, (2, 3, 4)))
    assert_equal(arr.shape, shape)


def test_reshaped_is_proxy():
    shape = (1, 2, 3, 4)
    hdr = FunkyHeader(shape)
    bio = BytesIO()
    prox = ArrayProxy(bio, hdr)
    assert_true(isinstance(prox.reshape((2, 3, 4)), ArrayProxy))
    minus1 = prox.reshape((2, -1, 4))
    assert_true(isinstance(minus1, ArrayProxy))
    assert_equal(minus1.shape, (2, 3, 4))
    assert_raises(ValueError, prox.reshape, (-1, -1, 4))
    assert_raises(ValueError, prox.reshape, (2, 3, 5))
    assert_raises(ValueError, prox.reshape, (2, -1, 5))


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
    # Whether ufunc on memmap return memmap
    viral_memmap = memmap_after_ufunc()
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
                             viral_memmap or unscaled_really_mmap)
                assert_equal(back_is_mmap,
                             viral_memmap or scaled_really_mmap)
                if scaled_really_mmap:
                    assert_equal(back_data.mode, expected_mode)
            del prox, back_data
            # Check that mmap is keyword-only
            assert_raises(TypeError, proxy_class, fname, hdr, True)
            # Check invalid values raise error
            assert_raises(ValueError, proxy_class, fname, hdr, mmap='rw')
            assert_raises(ValueError, proxy_class, fname, hdr, mmap='r+')


# An image opener class which counts how many instances of itself have been
# created
class CountingImageOpener(ImageOpener):
    num_openers = 0
    def __init__(self, *args, **kwargs):
        super(CountingImageOpener, self).__init__(*args, **kwargs)
        CountingImageOpener.num_openers += 1


def _count_ImageOpeners(proxy, data, voxels):
    CountingImageOpener.num_openers = 0
    for i in range(voxels.shape[0]):
        x, y, z = [int(c) for c in voxels[i, :]]
        assert proxy[x, y, z] == x * 100 + y * 10 + z
    return CountingImageOpener.num_openers


def test_keep_file_open_true_false_invalid():
    # Test the behaviour of the keep_file_open __init__ flag, when it is set to
    # True or False.
    CountingImageOpener.num_openers = 0
    fname = 'testdata'
    dtype = np.float32
    data  = np.arange(1000, dtype=dtype).reshape((10, 10, 10))
    voxels = np.random.randint(0, 10, (10, 3))
    with InTemporaryDirectory():
        with open(fname, 'wb') as fobj:
            fobj.write(data.tostring(order='F'))
        # Test that ArrayProxy(keep_file_open=True) only creates one file
        # handle, and that ArrayProxy(keep_file_open=False) creates a file
        # handle on every data access.
        with mock.patch('nibabel.openers.ImageOpener', CountingImageOpener):
            proxy_no_kfp = ArrayProxy(fname, ((10, 10, 10), dtype),
                                      keep_file_open=False)
            assert not proxy_no_kfp._keep_file_open
            assert _count_ImageOpeners(proxy_no_kfp, data, voxels) == 10
            proxy_kfp = ArrayProxy(fname, ((10, 10, 10), dtype),
                                   keep_file_open=True)
            assert proxy_kfp._keep_file_open
            assert _count_ImageOpeners(proxy_kfp, data, voxels) == 1
            del proxy_kfp
            del proxy_no_kfp
        # Test that the keep_file_open flag has no effect if an open file
        # handle is passed in
        with open(fname, 'rb') as fobj:
            for kfo in (True, False, 'auto'):
                proxy = ArrayProxy(fobj, ((10, 10, 10), dtype),
                                   keep_file_open=kfo)
                assert proxy._keep_file_open is False
                for i in range(voxels.shape[0]):
                    x, y, z = [int(c) for c in voxels[i, :]]
                    assert proxy[x, y, z] == x * 100 + y * 10 + z
                    assert not fobj.closed
                del proxy
                assert not fobj.closed
        assert fobj.closed
        # Test invalid values of keep_file_open
        with assert_raises(ValueError):
            ArrayProxy(fname, ((10, 10, 10), dtype), keep_file_open=55)
        with assert_raises(ValueError):
            ArrayProxy(fname, ((10, 10, 10), dtype), keep_file_open='autob')
        with assert_raises(ValueError):
            ArrayProxy(fname, ((10, 10, 10), dtype), keep_file_open='cauto')


def test_keep_file_open_auto():
    # Test the behaviour of the keep_file_open __init__ flag, when it is set to
    # 'auto'.
    # if indexed_gzip is present, the ArrayProxy should persist its ImageOpener.
    # Otherwise the ArrayProxy should drop openers.
    dtype = np.float32
    data = np.arange(1000, dtype=dtype).reshape((10, 10, 10))
    voxels = np.random.randint(0, 10, (10, 3))
    with InTemporaryDirectory():
        fname  = 'testdata.gz'
        with gzip.open(fname, 'wb') as fobj:
            fobj.write(data.tostring(order='F'))
        # If have_indexed_gzip, then the arrayproxy should create one
        # ImageOpener
        with patch_indexed_gzip(True), \
             mock.patch('nibabel.openers.ImageOpener', CountingImageOpener):
            CountingImageOpener.num_openers = 0
            proxy = ArrayProxy(fname, ((10, 10, 10), dtype),
                               keep_file_open='auto')
            assert proxy._keep_file_open == 'auto'
            assert _count_ImageOpeners(proxy, data, voxels) == 1
        # If no have_indexed_gzip, then keep_file_open should be False
        with patch_indexed_gzip(False), \
             mock.patch('nibabel.openers.ImageOpener', CountingImageOpener):
            CountingImageOpener.num_openers = 0
            proxy = ArrayProxy(fname, ((10, 10, 10), dtype),
                               keep_file_open='auto')
            assert proxy._keep_file_open is False
            assert _count_ImageOpeners(proxy, data, voxels) == 10
        # If not a gzip file,  keep_file_open should be False
        fname  = 'testdata'
        with open(fname, 'wb') as fobj:
            fobj.write(data.tostring(order='F'))
        # regardless of whether indexed_gzip is present or not
        with patch_indexed_gzip(True), \
             mock.patch('nibabel.openers.ImageOpener', CountingImageOpener):
            CountingImageOpener.num_openers = 0
            proxy = ArrayProxy(fname, ((10, 10, 10), dtype),
                               keep_file_open='auto')
            assert proxy._keep_file_open is False
            assert _count_ImageOpeners(proxy, data, voxels) == 10
        with patch_indexed_gzip(False), \
             mock.patch('nibabel.openers.ImageOpener', CountingImageOpener):
            CountingImageOpener.num_openers = 0
            proxy = ArrayProxy(fname, ((10, 10, 10), dtype),
                               keep_file_open='auto')
            assert proxy._keep_file_open is False
            assert _count_ImageOpeners(proxy, data, voxels) == 10


@contextlib.contextmanager
def patch_keep_file_open_default(value):
    # Patch arrayproxy.KEEP_FILE_OPEN_DEFAULT with the given value
    with mock.patch('nibabel.arrayproxy.KEEP_FILE_OPEN_DEFAULT', value):
        yield


def test_keep_file_open_default():
    # Test the behaviour of the keep_file_open __init__ flag, when the
    # arrayproxy.KEEP_FILE_OPEN_DEFAULT value is changed
    dtype = np.float32
    data = np.arange(1000, dtype=dtype).reshape((10, 10, 10))
    with InTemporaryDirectory():
        fname  = 'testdata.gz'
        with gzip.open(fname, 'wb') as fobj:
            fobj.write(data.tostring(order='F'))
        # If KEEP_FILE_OPEN_DEFAULT is False, ArrayProxy instances should
        # interpret keep_file_open as False
        with patch_keep_file_open_default(False):
            with patch_indexed_gzip(False):
                proxy = ArrayProxy(fname, ((10, 10, 10), dtype))
                assert proxy._keep_file_open is False
            with patch_indexed_gzip(True):
                proxy = ArrayProxy(fname, ((10, 10, 10), dtype))
                assert proxy._keep_file_open is False
        # If KEEP_FILE_OPEN_DEFAULT is True, ArrayProxy instances should
        # interpret keep_file_open as True
        with patch_keep_file_open_default(True):
            with patch_indexed_gzip(False):
                proxy = ArrayProxy(fname, ((10, 10, 10), dtype))
                assert proxy._keep_file_open is True
            with patch_indexed_gzip(True):
                proxy = ArrayProxy(fname, ((10, 10, 10), dtype))
                assert proxy._keep_file_open is True
        # If KEEP_FILE_OPEN_DEFAULT is auto, ArrayProxy instances should
        # interpret it as auto if indexed_gzip is present, False otherwise.
        with patch_keep_file_open_default('auto'):
            with patch_indexed_gzip(False):
                proxy = ArrayProxy(fname, ((10, 10, 10), dtype))
                assert proxy._keep_file_open is False
            with patch_indexed_gzip(True):
                proxy = ArrayProxy(fname, ((10, 10, 10), dtype))
                assert proxy._keep_file_open == 'auto'
        # KEEP_FILE_OPEN_DEFAULT=any other value should cuse an error to be
        # raised
        with patch_keep_file_open_default('badvalue'):
            assert_raises(ValueError,  ArrayProxy, fname, ((10, 10, 10),
                                                           dtype))
        with patch_keep_file_open_default(None):
            assert_raises(ValueError,  ArrayProxy, fname, ((10, 10, 10),
                                                           dtype))


def test_pickle_lock():
    # Test that ArrayProxy can be pickled, and that thread lock is created

    def islock(l):
        # isinstance doesn't work on threading.Lock?
        return hasattr(l, 'acquire') and hasattr(l, 'release')

    proxy = ArrayProxy('dummyfile', ((10, 10, 10), np.float32))
    assert islock(proxy._lock)
    pickled = pickle.dumps(proxy)
    unpickled = pickle.loads(pickled)
    assert islock(unpickled._lock)
    assert proxy._lock is not unpickled._lock
