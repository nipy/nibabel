# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Test for volumeutils module '''
from __future__ import with_statement
from ..py3k import BytesIO, asbytes, ZEROB
import tempfile

import numpy as np

from ..tmpdirs import InTemporaryDirectory

from ..volumeutils import (array_from_file,
                           array_to_file,
                           calculate_scale,
                           can_cast,
                           write_zeros,
                           apply_read_scaling,
                           _inter_type,
                           working_type,
                           best_write_scale_ftype,
                           better_float_of,
                           int_scinter_ftype,
                           allopen,
                           make_dt_codes,
                           native_code,
                           shape_zoom_affine,
                           rec2dict)

from ..casting import (floor_log2, type_info, best_float, OK_FLOATS)

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_equal, assert_raises

from ..testing import assert_dt_equal

#: convenience variables for numpy types
FLOAT_TYPES = np.sctypes['float']
CFLOAT_TYPES = np.sctypes['complex'] + FLOAT_TYPES
IUINT_TYPES = np.sctypes['int'] + np.sctypes['uint']
NUMERIC_TYPES = CFLOAT_TYPES + IUINT_TYPES


def test_array_from_file():
    shape = (2,3,4)
    dtype = np.dtype(np.float32)
    in_arr = np.arange(24, dtype=dtype).reshape(shape)
    # Check on string buffers
    offset = 0
    assert_true(buf_chk(in_arr, BytesIO(), None, offset))
    offset = 10
    assert_true(buf_chk(in_arr, BytesIO(), None, offset))
    # check on real file
    fname = 'test.bin'
    with InTemporaryDirectory():
        # fortran ordered
        out_buf = open(fname, 'wb')
        in_buf = open(fname, 'rb')
        assert_true(buf_chk(in_arr, out_buf, in_buf, offset))
        # Drop offset to check that shape's not coming from file length
        out_buf.seek(0)
        in_buf.seek(0)
        offset = 5
        assert_true(buf_chk(in_arr, out_buf, in_buf, offset))
        del out_buf, in_buf
    # Make sure empty shape, and zero length, give empty arrays
    arr = array_from_file((), np.dtype('f8'), BytesIO())
    assert_equal(len(arr), 0)
    arr = array_from_file((0,), np.dtype('f8'), BytesIO())
    assert_equal(len(arr), 0)
    # Check error from small file
    assert_raises(IOError, array_from_file,
                        shape, dtype, BytesIO())
    # check on real file
    fd, fname = tempfile.mkstemp()
    with InTemporaryDirectory():
        open(fname, 'wb').write(asbytes('1'))
        in_buf = open(fname, 'rb')
        # For windows this will raise a WindowsError from mmap, Unices
        # appear to raise an IOError
        assert_raises(Exception, array_from_file,
                            shape, dtype, in_buf)
        del in_buf


def buf_chk(in_arr, out_buf, in_buf, offset):
    ''' Write contents of in_arr into fileobj, read back, check same '''
    instr = asbytes(' ') * offset + in_arr.tostring(order='F')
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


def test_array_to_file():
    arr = np.arange(10).reshape(5,2)
    str_io = BytesIO()
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
                assert_array_almost_equal(arr, data_back)


def test_a2f_intercept_scale():
    arr = np.array([0.0, 1.0, 2.0])
    str_io = BytesIO()
    # intercept
    data_back = write_return(arr, str_io, np.float64, 0, 1.0)
    assert_array_equal(data_back, arr-1)
    # scaling
    data_back = write_return(arr, str_io, np.float64, 0, 1.0, 2.0)
    assert_array_equal(data_back, (arr-1) / 2.0)


def test_a2f_upscale():
    # Test working type scales with needed range
    info = type_info(np.float32)
    # Test values discovered from stress testing.  The largish value (2**115)
    # overflows to inf after the intercept is subtracted, using float32 as the
    # working precision.  The difference between inf and this value is lost.
    arr = np.array([[info['min'], 2**115, info['max']]], dtype=np.float32)
    slope = np.float32(2**121)
    inter = info['min']
    str_io = BytesIO()
    # We need to provide mn, mx for function to be able to calculate upcasting
    array_to_file(arr, str_io, np.uint8, intercept=inter, divslope=slope,
                  mn = info['min'], mx = info['max'])
    raw = array_from_file(arr.shape, np.uint8, str_io)
    back = apply_read_scaling(raw, slope, inter)
    top = back - arr
    score = np.abs(top / arr)
    assert_true(np.all(score < 10))


def test_a2f_min_max():
    str_io = BytesIO()
    for in_dt in (np.float32, np.int8):
        for out_dt in (np.float32, np.int8):
            arr = np.arange(4, dtype=in_dt)
            # min thresholding
            data_back = write_return(arr, str_io, out_dt, 0, 0, 1, 1)
            assert_array_equal(data_back, [1, 1, 2, 3])
            # max thresholding
            data_back = write_return(arr, str_io, out_dt, 0, 0, 1, None, 2)
            assert_array_equal(data_back, [0, 1, 2, 2])
            # min max thresholding
            data_back = write_return(arr, str_io, out_dt, 0, 0, 1, 1, 2)
            assert_array_equal(data_back, [1, 1, 2, 2])
    # Check that works OK with scaling and intercept
    arr = np.arange(4, dtype=np.float32)
    data_back = write_return(arr, str_io, np.int, 0, -1, 0.5, 1, 2)
    assert_array_equal(data_back * 0.5 - 1, [1, 1, 2, 2])
    # Even when scaling is negative
    data_back = write_return(arr, str_io, np.int, 0, 1, -0.5, 1, 2)
    assert_array_equal(data_back * -0.5 + 1, [1, 1, 2, 2])


def test_a2f_order():
    ndt = np.dtype(np.float)
    arr = np.array([0.0, 1.0, 2.0])
    str_io = BytesIO()
    # order makes no difference in 1D case
    data_back = write_return(arr, str_io, ndt, order='C')
    assert_array_equal(data_back, [0.0, 1.0, 2.0])
    # but does in the 2D case
    arr = np.array([[0.0, 1.0],[2.0, 3.0]])
    data_back = write_return(arr, str_io, ndt, order='F')
    assert_array_equal(data_back, arr)
    data_back = write_return(arr, str_io, ndt, order='C')
    assert_array_equal(data_back, arr.T)


def test_a2f_nan2zero():
    ndt = np.dtype(np.float)
    str_io = BytesIO()
    # nans set to 0 for integer output case, not float
    arr = np.array([[np.nan, 0],[0, np.nan]])
    data_back = write_return(arr, str_io, ndt) # float, thus no effect
    assert_array_equal(data_back, arr)
    # True is the default, but just to show its possible
    data_back = write_return(arr, str_io, ndt, nan2zero=True)
    assert_array_equal(data_back, arr)
    data_back = write_return(arr, str_io,
                             np.dtype(np.int64), nan2zero=True)
    assert_array_equal(data_back, [[0, 0],[0, 0]])
    # otherwise things get a bit weird; tidied here
    # How weird?  Look at arr.astype(np.int64)
    data_back = write_return(arr, str_io,
                             np.dtype(np.int64), nan2zero=False)
    assert_array_equal(data_back, arr.astype(np.int64))


def test_a2f_offset():
    # check that non-zero file offset works
    arr = np.array([[0.0, 1.0],[2.0, 3.0]])
    str_io = BytesIO()
    str_io.write(asbytes('a') * 42)
    array_to_file(arr, str_io, np.float, 42)
    data_back = array_from_file(arr.shape, np.float, str_io, 42)
    assert_array_equal(data_back, arr.astype(np.float))
    # And that offset=None respected
    str_io.truncate(22)
    str_io.seek(22)
    array_to_file(arr, str_io, np.float, None)
    data_back = array_from_file(arr.shape, np.float, str_io, 22)
    assert_array_equal(data_back, arr.astype(np.float))


def test_a2f_dtype_default():
    # that default dtype is input dtype
    arr = np.array([[0.0, 1.0],[2.0, 3.0]])
    str_io = BytesIO()
    array_to_file(arr.astype(np.int16), str_io)
    data_back = array_from_file(arr.shape, np.int16, str_io)
    assert_array_equal(data_back, arr.astype(np.int16))


def test_a2f_zeros():
    # Check that, if there is no valid data, we get zeros
    arr = np.array([[0.0, 1.0],[2.0, 3.0]])
    str_io = BytesIO()
    # With slope=None signal
    array_to_file(arr + np.inf, str_io, np.int32, 0, 0.0, None)
    data_back = array_from_file(arr.shape, np.int32, str_io)
    assert_array_equal(data_back, np.zeros(arr.shape))
    # With  mn, mx = 0 signal
    array_to_file(arr, str_io, np.int32, 0, 0.0, 1.0, 0, 0)
    data_back = array_from_file(arr.shape, np.int32, str_io)
    assert_array_equal(data_back, np.zeros(arr.shape))
    # With  mx < mn signal
    array_to_file(arr, str_io, np.int32, 0, 0.0, 1.0, 4, 2)
    data_back = array_from_file(arr.shape, np.int32, str_io)
    assert_array_equal(data_back, np.zeros(arr.shape))


def test_a2f_big_scalers():
    # Check that clip works even for overflowing scalers / data
    info = type_info(np.float32)
    arr = np.array([info['min'], np.nan, info['max']], dtype=np.float32)
    str_io = BytesIO()
    # Intercept causes overflow - does routine scale correctly?
    array_to_file(arr, str_io, np.int8, intercept=np.float32(2**120))
    data_back = array_from_file(arr.shape, np.int8, str_io)
    assert_array_equal(data_back, [-128, 0, 127])
    # Scales also if mx, mn specified?
    str_io.seek(0)
    array_to_file(arr, str_io, np.int8, mn=info['min'], mx=info['max'],
                  intercept=np.float32(2**120))
    data_back = array_from_file(arr.shape, np.int8, str_io)
    assert_array_equal(data_back, [-128, 0, 127])
    # And if slope causes overflow?
    str_io.seek(0)
    array_to_file(arr, str_io, np.int8, divslope=np.float32(0.5))
    data_back = array_from_file(arr.shape, np.int8, str_io)
    assert_array_equal(data_back, [-128, 0, 127])
    # with mn, mx specified?
    str_io.seek(0)
    array_to_file(arr, str_io, np.int8, mn=info['min'], mx=info['max'],
                  divslope=np.float32(0.5))
    data_back = array_from_file(arr.shape, np.int8, str_io)
    assert_array_equal(data_back, [-128, 0, 127])


def write_return(data, fileobj, out_dtype, *args, **kwargs):
    fileobj.truncate(0)
    array_to_file(data, fileobj, out_dtype, *args, **kwargs)
    data = array_from_file(data.shape, out_dtype, fileobj)
    return data


def test_apply_scaling():
    # Null scaling, same array returned
    arr = np.zeros((3,), dtype=np.int16)
    assert_true(apply_read_scaling(arr) is arr)
    assert_true(apply_read_scaling(arr, np.float64(1.0)) is arr)
    assert_true(apply_read_scaling(arr, inter=np.float64(0)) is arr)
    f32, f64 = np.float32, np.float64
    f32_arr = np.zeros((1,), dtype=f32)
    i16_arr = np.zeros((1,), dtype=np.int16)
    # Check float upcast (not the normal numpy scalar rule)
    # This is the normal rule - no upcast from scalar
    assert_equal((f32_arr * f64(1)).dtype, np.float32)
    assert_equal((f32_arr + f64(1)).dtype, np.float32)
    # The function does upcast though
    ret = apply_read_scaling(np.float32(0), np.float64(2))
    assert_equal(ret.dtype, np.float64)
    ret = apply_read_scaling(np.float32(0), inter=np.float64(2))
    assert_equal(ret.dtype, np.float64)
    # Check integer inf upcast
    big = f32(type_info(f32)['max'])
    # Normally this would not upcast
    assert_equal((i16_arr * big).dtype, np.float32)
    # An equivalent case is a little hard to find for the intercept
    nmant_32 = type_info(np.float32)['nmant']
    big_delta = np.float32(2**(floor_log2(big)-nmant_32))
    assert_equal((i16_arr * big_delta + big).dtype, np.float32)
    # Upcasting does occur with this routine
    assert_equal(apply_read_scaling(i16_arr, big).dtype, np.float64)
    assert_equal(apply_read_scaling(i16_arr, big_delta, big).dtype, np.float64)
    # If float32 passed, no overflow, float32 returned
    assert_equal(apply_read_scaling(np.int8(0), f32(-1.0), f32(0.0)).dtype,
                 np.float32)
    # float64 passed, float64 returned
    assert_equal(apply_read_scaling(np.int8(0), -1.0, 0.0).dtype, np.float64)
    # float32 passed, overflow, float64 returned
    assert_equal(apply_read_scaling(np.int8(0), f32(1e38), f32(0.0)).dtype,
                 np.float64)
    assert_equal(apply_read_scaling(np.int8(0), f32(-1e38), f32(0.0)).dtype,
                 np.float64)
    # Test that integer casting during read scaling works
    assert_dt_equal(apply_read_scaling(i16_arr, 1.0, 1.0).dtype, np.int32)
    assert_dt_equal(apply_read_scaling(
        np.zeros((1,), dtype=np.int32), 1.0, 1.0).dtype, np.int64)
    assert_dt_equal(apply_read_scaling(
        np.zeros((1,), dtype=np.int64), 1.0, 1.0).dtype, best_float())


def test__inter_type():
    # Test routine to get intercept type
    bf = best_float()
    for in_type, inter, out_type, exp_out in (
        (np.int8, 0, None, np.int8),
        (np.int8, 0, np.int8, np.int8),
        (np.int8, 1, None, np.int16),
        (np.int8, 1, np.int8, bf),
        (np.int8, 1, np.int16, np.int16),
        (np.uint8, 0, None, np.uint8),
        (np.uint8, 1, None, np.uint16),
        (np.uint8, -1, None, np.int16),
        (np.int16, 1, None, np.int32),
        (np.uint16, 0, None, np.uint16),
        (np.uint16, 1, None, np.uint32),
        (np.int32, 1, None, np.int64),
        (np.uint32, 1, None, np.uint64),
        (np.int64, 1, None, bf),
        (np.uint64, 1, None, bf),
    ):
        assert_dt_equal(_inter_type(in_type, inter, out_type), exp_out)
        # Check that casting is as expected
        A = np.zeros((1,), dtype=in_type)
        B = np.array([inter], dtype=exp_out)
        ApBt = (A + B).dtype.type
        assert_dt_equal(ApBt, exp_out)


def test_int_scinter():
    # Finding float type needed for applying scale, offset to ints
    assert_equal(int_scinter_ftype(np.int8, 1.0, 0.0), np.float32)
    assert_equal(int_scinter_ftype(np.int8, -1.0, 0.0), np.float32)
    assert_equal(int_scinter_ftype(np.int8, 1e38, 0.0), np.float64)
    assert_equal(int_scinter_ftype(np.int8, -1e38, 0.0), np.float64)


def test_working_type():
    # Which type do input types with slope and inter cast to in numpy?
    # Wrapper function because we need to use the dtype str for comparison.  We
    # need this because of the very confusing np.int32 != np.intp (on 32 bit).
    def wt(*args, **kwargs):
        return np.dtype(working_type(*args, **kwargs)).str
    d1 = np.atleast_1d
    for in_type in NUMERIC_TYPES:
        in_ts = np.dtype(in_type).str
        assert_equal(wt(in_type), in_ts)
        assert_equal(wt(in_type, 1, 0), in_ts)
        assert_equal(wt(in_type, 1.0, 0.0), in_ts)
        in_val = d1(in_type(0))
        for slope_type in NUMERIC_TYPES:
            sl_val = slope_type(1) # no scaling, regardless of type
            assert_equal(wt(in_type, sl_val, 0.0), in_ts)
            sl_val = slope_type(2) # actual scaling
            out_val = in_val / d1(sl_val)
            assert_equal(wt(in_type, sl_val), out_val.dtype.str)
            for inter_type in NUMERIC_TYPES:
                i_val = inter_type(0) # no scaling, regardless of type
                assert_equal(wt(in_type, 1, i_val), in_ts)
                i_val = inter_type(1) # actual scaling
                out_val = in_val - d1(i_val)
                assert_equal(wt(in_type, 1, i_val), out_val.dtype.str)
                # Combine scaling and intercept
                out_val = (in_val - d1(i_val)) / d1(sl_val)
                assert_equal(wt(in_type, sl_val, i_val), out_val.dtype.str)
    # Confirm that type codes and dtypes work as well
    f32s = np.dtype(np.float32).str
    assert_equal(wt('f4', 1, 0), f32s)
    assert_equal(wt(np.dtype('f4'), 1, 0), f32s)


def test_better_float():
    # Better float function
    def check_against(f1, f2):
        return f1 if FLOAT_TYPES.index(f1) >= FLOAT_TYPES.index(f2) else f2
    for first in FLOAT_TYPES:
        for other in IUINT_TYPES + np.sctypes['complex']:
            assert_equal(better_float_of(first, other), first)
            assert_equal(better_float_of(other, first), first)
            for other2 in IUINT_TYPES + np.sctypes['complex']:
                assert_equal(better_float_of(other, other2), np.float32)
                assert_equal(better_float_of(other, other2, np.float64),
                             np.float64)
        for second in FLOAT_TYPES:
            assert_equal(better_float_of(first, second),
                         check_against(first, second))
    # Check codes and dtypes work
    assert_equal(better_float_of('f4', 'f8', 'f4'), np.float64)
    assert_equal(better_float_of('i4', 'i8', 'f8'), np.float64)


def test_best_write_scale_ftype():
    # Test best write scaling type
    # Types return better of (default, array type) unless scale overflows.
    # Return float type cannot be less capable than the input array type
    for dtt in IUINT_TYPES + FLOAT_TYPES:
        arr = np.arange(10, dtype=dtt)
        assert_equal(best_write_scale_ftype(arr, 1, 0),
                     better_float_of(dtt, np.float32))
        assert_equal(best_write_scale_ftype(arr, 1, 0, np.float64),
                     better_float_of(dtt, np.float64))
        assert_equal(best_write_scale_ftype(arr, np.float32(2), 0),
                     better_float_of(dtt, np.float32))
        assert_equal(best_write_scale_ftype(arr, 1, np.float32(1)),
                     better_float_of(dtt, np.float32))
    # Overflowing ints with scaling results in upcast
    best_vals = ((np.float32, np.float64),)
    if np.longdouble in OK_FLOATS:
        best_vals += ((np.float64, np.longdouble),)
    for lower_t, higher_t in best_vals:
        # Information on this float
        L_info = type_info(lower_t)
        t_max = L_info['max']
        nmant = L_info['nmant'] # number of significand digits
        big_delta = lower_t(2**(floor_log2(t_max) - nmant)) # delta below max
        # Even large values that don't overflow don't change output
        arr = np.array([0, t_max], dtype=lower_t)
        assert_equal(best_write_scale_ftype(arr, 1, 0), lower_t)
        # Scaling > 1 reduces output values, so no upcast needed
        assert_equal(best_write_scale_ftype(arr, lower_t(1.01), 0), lower_t)
        # Scaling < 1 increases values, so upcast may be needed (and is here)
        assert_equal(best_write_scale_ftype(arr, lower_t(0.99), 0), higher_t)
        # Large minus offset on large array can cause upcast
        assert_equal(best_write_scale_ftype(arr, 1, -big_delta/2.01), lower_t)
        assert_equal(best_write_scale_ftype(arr, 1, -big_delta/2.0), higher_t)
        # With infs already in input, default type returns
        arr[0] = np.inf
        assert_equal(best_write_scale_ftype(arr, lower_t(0.5), 0), lower_t)
        arr[0] = -np.inf
        assert_equal(best_write_scale_ftype(arr, lower_t(0.5), 0), lower_t)


def test_can_cast():
    tests = ((np.float32, np.float32, True, True, True),
             (np.float64, np.float32, True, True, True),
             (np.complex128, np.float32, False, False, False),
             (np.float32, np.complex128, True, True, True),
             (np.float32, np.uint8, False, True, True),
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
        assert_equal(def_res, can_cast(intype, outtype))
        assert_equal(scale_res, can_cast(intype, outtype, False, True))
        assert_equal(all_res, can_cast(intype, outtype, True, True))


def test_write_zeros():
    bio = BytesIO()
    write_zeros(bio, 10000)
    assert_equal(bio.getvalue(), ZEROB*10000)
    bio.seek(0)
    bio.truncate(0)
    write_zeros(bio, 10000, 256)
    assert_equal(bio.getvalue(), ZEROB*10000)
    bio.seek(0)
    bio.truncate(0)
    write_zeros(bio, 200, 256)
    assert_equal(bio.getvalue(), ZEROB*200)


def test_allopen():
    # Test default mode is 'rb'
    fobj = allopen(__file__)
    assert_equal(fobj.mode, 'rb')
    # That we can set it
    fobj = allopen(__file__, 'r')
    assert_equal(fobj.mode, 'r')
    # with keyword arguments
    fobj = allopen(__file__, mode='r')
    assert_equal(fobj.mode, 'r')
    # fileobj returns fileobj
    sobj = BytesIO()
    fobj = allopen(sobj)
    assert_true(fobj is sobj)
    # mode is gently ignored
    fobj = allopen(sobj, mode='r')


def test_shape_zoom_affine():
    shape = (3, 5, 7)
    zooms = (3, 2, 1)
    res = shape_zoom_affine(shape, zooms)
    exp = np.array([[-3.,  0.,  0.,  3.],
                    [ 0.,  2.,  0., -4.],
                    [ 0.,  0.,  1., -3.],
                    [ 0.,  0.,  0.,  1.]])
    assert_array_almost_equal(res, exp)
    res = shape_zoom_affine((3, 5), (3, 2))
    exp = np.array([[-3.,  0.,  0.,  3.],
                    [ 0.,  2.,  0., -4.],
                    [ 0.,  0.,  1., -0.],
                    [ 0.,  0.,  0.,  1.]])
    assert_array_almost_equal(res, exp)
    res = shape_zoom_affine(shape, zooms, False)
    exp = np.array([[ 3.,  0.,  0., -3.],
                    [ 0.,  2.,  0., -4.],
                    [ 0.,  0.,  1., -3.],
                    [ 0.,  0.,  0.,  1.]])
    assert_array_almost_equal(res, exp)


def test_rec2dict():
    r = np.zeros((), dtype = [('x', 'i4'), ('s', 'S10')])
    d = rec2dict(r)
    assert_equal(d, {'x': 0, 's': asbytes('')})


def test_dtypes():
    # numpy - at least up to 1.5.1 - has odd behavior for hashing -
    # specifically:
    # In [9]: hash(dtype('<f4')) == hash(dtype('<f4').newbyteorder('<'))
    # Out[9]: False
    # In [10]: dtype('<f4') == dtype('<f4').newbyteorder('<')
    # Out[10]: True
    # where '<' is the native byte order
    dt_defs = ((16, 'float32', np.float32),)
    dtr = make_dt_codes(dt_defs)
    # check we have the fields we were expecting
    assert_equal(dtr.value_set(), set((16,)))
    assert_equal(dtr.fields, ('code', 'label', 'type',
                              'dtype', 'sw_dtype'))
    # These of course should pass regardless of dtype
    assert_equal(dtr[np.float32], 16)
    assert_equal(dtr['float32'], 16)
    # These also pass despite dtype issue
    assert_equal(dtr[np.dtype(np.float32)], 16)
    assert_equal(dtr[np.dtype('f4')], 16)
    assert_equal(dtr[np.dtype('f4').newbyteorder('S')], 16)
    # But this one used to fail
    assert_equal(dtr[np.dtype('f4').newbyteorder(native_code)], 16)
    # Check we can pass in niistring as well
    dt_defs = ((16, 'float32', np.float32, 'ASTRING'),)
    dtr = make_dt_codes(dt_defs)
    assert_equal(dtr[np.dtype('f4').newbyteorder('S')], 16)
    assert_equal(dtr.value_set(), set((16,)))
    assert_equal(dtr.fields, ('code', 'label', 'type', 'niistring',
                              'dtype', 'sw_dtype'))
    assert_equal(dtr.niistring[16], 'ASTRING')
    # And that unequal elements raises error
    dt_defs = ((16, 'float32', np.float32, 'ASTRING'),
               (16, 'float32', np.float32))
    assert_raises(ValueError, make_dt_codes, dt_defs)
    # And that 2 or 5 elements raises error
    dt_defs = ((16, 'float32'),)
    assert_raises(ValueError, make_dt_codes, dt_defs)
    dt_defs = ((16, 'float32', np.float32, 'ASTRING', 'ANOTHERSTRING'),)
    assert_raises(ValueError, make_dt_codes, dt_defs)
