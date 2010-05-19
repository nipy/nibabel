''' Test for volumeutils module '''
from __future__ import with_statement
import os
from StringIO import StringIO
import tempfile

import numpy as np

from nibabel.tmpdirs import InTemporaryDirectory

from nibabel.volumeutils import array_from_file, \
    array_to_file, calculate_scale, scale_min_max, can_cast

from numpy.testing import assert_array_almost_equal, \
    assert_array_equal

from nose.tools import assert_true, assert_equal, assert_raises

from nibabel.testing import parametric


@parametric
def test_array_from_file():
    shape = (2,3,4)
    dtype = np.dtype(np.float32)
    in_arr = np.arange(24, dtype=dtype).reshape(shape)
    # Check on string buffers
    offset = 0
    yield assert_true(buf_chk(in_arr, StringIO(), None, offset))
    offset = 10
    yield assert_true(buf_chk(in_arr, StringIO(), None, offset))
    # check on real file
    fname = 'test.bin'
    with InTemporaryDirectory() as tmpdir:
        # fortran ordered
        out_buf = file(fname, 'wb')
        in_buf = file(fname, 'rb')
        yield assert_true(buf_chk(in_arr, out_buf, in_buf, offset))
        # Drop offset to check that shape's not coming from file length
        out_buf.seek(0)
        in_buf.seek(0)
        offset = 5
        yield assert_true(buf_chk(in_arr, out_buf, in_buf, offset))
        del out_buf, in_buf
    # Make sure empty shape, and zero length, give empty arrays
    arr = array_from_file((), np.dtype('f8'), StringIO())
    yield assert_equal(len(arr), 0)
    arr = array_from_file((0,), np.dtype('f8'), StringIO())
    yield assert_equal(len(arr), 0)
    # Check error from small file
    yield assert_raises(IOError, array_from_file,
                        shape, dtype, StringIO())
    # check on real file
    fd, fname = tempfile.mkstemp()
    with InTemporaryDirectory():
        open(fname, 'wb').write('1')
        in_buf = open(fname, 'rb')
        # For windows this will raise a WindowsError from mmap, Unices
        # appear to raise an IOError
        yield assert_raises(Exception, array_from_file,
                            shape, dtype, in_buf)
        del in_buf


def buf_chk(in_arr, out_buf, in_buf, offset):
    ''' Write contents of in_arr into fileobj, read back, check same '''
    instr = ' ' * offset + in_arr.tostring(order='F')
    out_buf.write(instr)
    out_buf.flush()
    if in_buf is None: # we're using in_buf from out_buf
        out_buf.seek(0)
        in_buf = out_buf
    arr = array_from_file(
        in_arr.shape,
        in_arr.dtype,
        in_buf,
        offset)
    return np.allclose(in_arr, arr)


@parametric
def test_array_to_file():
    arr = np.arange(10).reshape(5,2)
    str_io = StringIO()
    for tp in (np.uint64, np.float, np.complex):
        dt = np.dtype(tp)
        for code in '<>':
            ndt = dt.newbyteorder(code)
            for allow_intercept in (True, False):
                scale, intercept, mn, mx = calculate_scale(arr,
                                                           ndt,
                                                           allow_intercept)
                data_back = write_return(arr, str_io, ndt,
                                         0, intercept, scale)
                yield assert_array_almost_equal(arr, data_back)
    ndt = np.dtype(np.float)
    arr = np.array([0.0, 1.0, 2.0])
    # intercept
    data_back = write_return(arr, str_io, ndt, 0, 1.0)
    yield assert_array_equal(data_back, arr-1)
    # scaling
    data_back = write_return(arr, str_io, ndt, 0, 1.0, 2.0)
    yield assert_array_equal(data_back, (arr-1) / 2.0)
    # min thresholding
    data_back = write_return(arr, str_io, ndt, 0, 0.0, 1.0, 1.0)
    yield assert_array_equal(data_back, [1.0, 1.0, 2.0])
    # max thresholding
    data_back = write_return(arr, str_io, ndt, 0, 0.0, 1.0, 0.0, 1.0)
    yield assert_array_equal(data_back, [0.0, 1.0, 1.0])
    # order makes not difference in 1D case
    data_back = write_return(arr, str_io, ndt, order='C')
    yield assert_array_equal(data_back, [0.0, 1.0, 2.0])
    # but does in the 2D case
    arr = np.array([[0.0, 1.0],[2.0, 3.0]])
    data_back = write_return(arr, str_io, ndt, order='F')
    yield assert_array_equal(data_back, arr)
    data_back = write_return(arr, str_io, ndt, order='C')
    yield assert_array_equal(data_back, arr.T)
    # nans set to 0 for integer output case, not float
    arr = np.array([[np.nan, 0],[0, np.nan]])
    data_back = write_return(arr, str_io, ndt) # float, thus no effect
    yield assert_array_equal(data_back, arr)
    # True is the default, but just to show its possible
    data_back = write_return(arr, str_io, ndt, nan2zero=True) 
    yield assert_array_equal(data_back, arr)
    data_back = write_return(arr, str_io,
                             np.dtype(np.int64), nan2zero=True) 
    yield assert_array_equal(data_back, [[0, 0],[0, 0]])
    # otherwise things get a bit weird; tidied here
    # How weird?  Look at arr.astype(np.int64)
    data_back = write_return(arr, str_io,
                             np.dtype(np.int64), nan2zero=False)
    yield assert_array_equal(data_back, arr.astype(np.int64))
    # check that non-zero file offset works
    arr = np.array([[0.0, 1.0],[2.0, 3.0]])
    str_io = StringIO()
    str_io.write('a' * 42)
    array_to_file(arr, str_io, np.float, 42)
    data_back = array_from_file(arr.shape, np.float, str_io, 42)
    yield assert_array_equal(data_back, arr.astype(np.float))
    # that default dtype is input dtype
    str_io = StringIO()
    array_to_file(arr.astype(np.int16), str_io)
    data_back = array_from_file(arr.shape, np.int16, str_io)
    yield assert_array_equal(data_back, arr.astype(np.int16))
    # that, if there is no valid data, we get zeros
    str_io = StringIO()
    array_to_file(arr + np.inf, str_io, np.int32, 0, 0.0, None)
    data_back = array_from_file(arr.shape, np.int32, str_io)
    yield assert_array_equal(data_back, np.zeros(arr.shape))
    
    
def write_return(data, fileobj, out_dtype, *args, **kwargs):
    fileobj.truncate(0)
    array_to_file(data, fileobj, out_dtype, *args, **kwargs)
    data = array_from_file(data.shape, out_dtype, fileobj)
    return data


def test_scale_min_max():
    mx_dt = np.maximum_sctype(np.float)
    for tp in np.sctypes['uint'] + np.sctypes['int']:
        info = np.iinfo(tp)
        # Need to pump up to max fp type to contain python longs
        imin = np.array(info.min, dtype=mx_dt)
        imax = np.array(info.max, dtype=mx_dt)
        value_pairs = (
            (0, imax),
            (imin, 0),
            (imin, imax),
            (1, 10),
            (-1, -1),
            (1, 1),
            (-10, -1),
            (-100, 10))
        for mn, mx in value_pairs:
            # with intercept
            scale, inter = scale_min_max(mn, mx, tp, True)
            if mx-mn:
                yield assert_array_almost_equal, (mx-inter) / scale, imax
                yield assert_array_almost_equal, (mn-inter) / scale, imin
            else:
                yield assert_equal, (scale, inter), (1.0, mn)
            # without intercept
            if imin == 0 and mn < 0 and mx > 0:
                yield (assert_raises, ValueError,
                       scale_min_max, mn, mx, tp, False)
                continue
            scale, inter = scale_min_max(mn, mx, tp, False)
            yield assert_equal, inter, 0.0
            if mn == 0 and mx == 0:
                yield assert_equal, scale, 1.0
                continue
            sc_mn = mn / scale
            sc_mx = mx / scale
            yield assert_true, sc_mn >= imin
            yield assert_true, sc_mx <= imax
            if imin == 0:
                if mx > 0: # numbers all +ve
                    yield assert_array_almost_equal, mx / scale, imax
                else: # numbers all -ve
                    yield assert_array_almost_equal, mn / scale, imax
                continue
            if abs(mx) >= abs(mn):
                yield assert_array_almost_equal, mx / scale, imax
            else:
                yield assert_array_almost_equal, mn / scale, imin


def test_can_cast():
    tests = ((np.float32, np.float32, True, True, True),
             (np.float64, np.float32, True, True, True),
             (np.complex128, np.float32, False, False, False),
             (np.float32, np.complex128, True, True, True),
             (np.uint32, np.complex128, True, True, True),
             (np.int64, np.float32, True, True, True),
             (np.complex128, np.int16, False, False, False),
             (np.float32, np.int16, False, True, True),
             (np.uint8, np.int16, True, True, True),
             (np.uint16, np.int16, False, True, True),
             (np.int16, np.uint16, False, False, True),
             (np.int8, np.uint16, False, False, True),
             (np.uint16, np.uint8, False, True, True),
             )
    for intype, outtype, def_res, scale_res, all_res in tests:
        yield assert_equal, def_res, can_cast(intype, outtype)
        yield assert_equal, scale_res, can_cast(intype, outtype, False, True)
        yield assert_equal, all_res, can_cast(intype, outtype, True, True)
        
        
