# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Test for volumeutils module '''
from __future__ import division

from os.path import exists

from ..externals.six import BytesIO
import tempfile
import warnings
import functools
import itertools

import numpy as np

from ..tmpdirs import InTemporaryDirectory

from ..volumeutils import (array_from_file,
                           array_to_file,
                           allopen, # for backwards compatibility
                           BinOpener,
                           fname_ext_ul_case,
                           calculate_scale,
                           can_cast,
                           write_zeros,
                           seek_tell,
                           apply_read_scaling,
                           working_type,
                           best_write_scale_ftype,
                           better_float_of,
                           int_scinter_ftype,
                           make_dt_codes,
                           native_code,
                           shape_zoom_affine,
                           rec2dict,
                           _dt_min_max,
                           _write_data,
                          )

from ..casting import (floor_log2, type_info, best_float, OK_FLOATS,
                       shared_range)

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_equal, assert_raises

from ..testing import assert_dt_equal, assert_allclose_safely

#: convenience variables for numpy types
FLOAT_TYPES = np.sctypes['float']
COMPLEX_TYPES = np.sctypes['complex']
CFLOAT_TYPES = FLOAT_TYPES + COMPLEX_TYPES
INT_TYPES = np.sctypes['int']
IUINT_TYPES = INT_TYPES + np.sctypes['uint']
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
        open(fname, 'wb').write(b'1')
        in_buf = open(fname, 'rb')
        # For windows this will raise a WindowsError from mmap, Unices
        # appear to raise an IOError
        assert_raises(Exception, array_from_file,
                            shape, dtype, in_buf)
        del in_buf


def buf_chk(in_arr, out_buf, in_buf, offset):
    ''' Write contents of in_arr into fileobj, read back, check same '''
    instr = b' ' * offset + in_arr.tostring(order='F')
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
    # Test array-like
    str_io = BytesIO()
    array_to_file(arr.tolist(), str_io, float)
    data_back = array_from_file(arr.shape, float, str_io)
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
    # Check min and max thresholding of array to file
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
    # Check complex numbers
    arr = np.arange(4, dtype=np.complex64) + 100j
    data_back = write_return(arr, str_io, out_dt, 0, 0, 1, 1, 2)
    assert_array_equal(data_back, [1, 1, 2, 2])


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
    # True is the default, but just to show it's possible
    data_back = write_return(arr, str_io, ndt, nan2zero=True)
    assert_array_equal(data_back, arr)
    data_back = write_return(arr, str_io, np.int64, nan2zero=True)
    assert_array_equal(data_back, [[0, 0],[0, 0]])
    # otherwise things get a bit weird; tidied here
    # How weird?  Look at arr.astype(np.int64)
    data_back = write_return(arr, str_io, np.int64, nan2zero=False)
    assert_array_equal(data_back, arr.astype(np.int64))


def test_a2f_nan2zero_scaling():
    # Check that nan gets translated to the nearest equivalent to zero
    #
    # nan can be represented as zero of we can store (0 - intercept) / divslope
    # in the output data - because reading back the data as `stored_array  * divslope +
    # intercept` will reconstruct zeros for the nans in the original input.
    #
    # Check with array containing nan, matching array containing zero and
    # Array containing zero
    # Array values otherwise not including zero without scaling
    # Same with negative sign
    # Array values including zero before scaling but not after
    bio = BytesIO()
    for in_dt, out_dt, zero_in, inter in itertools.product(
        FLOAT_TYPES,
        IUINT_TYPES,
        (True, False),
        (0, -100)):
        in_info = np.finfo(in_dt)
        out_info = np.iinfo(out_dt)
        mx = min(in_info.max, out_info.max * 2., 2**32) + inter
        mn = 0 if zero_in or inter else 100
        vals = [np.nan] + [mn, mx]
        nan_arr = np.array(vals, dtype=in_dt)
        zero_arr = np.nan_to_num(nan_arr)
        back_nan = write_return(nan_arr, bio, np.int64, intercept=inter)
        back_zero = write_return(zero_arr, bio, np.int64, intercept=inter)
        assert_array_equal(back_nan, back_zero)


def test_a2f_offset():
    # check that non-zero file offset works
    arr = np.array([[0.0, 1.0],[2.0, 3.0]])
    str_io = BytesIO()
    str_io.write(b'a' * 42)
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
    arr = np.array([info['min'], 0, info['max']], dtype=np.float32)
    str_io = BytesIO()
    # Intercept causes overflow - does routine scale correctly?
    # We check whether the routine correctly clips extreme values.
    # We need nan2zero=False because we can't represent 0 in the input, given
    # the scaling and the output range.
    array_to_file(arr, str_io, np.int8, intercept=np.float32(2**120),
                  nan2zero=False)
    data_back = array_from_file(arr.shape, np.int8, str_io)
    assert_array_equal(data_back, [-128, -128, 127])
    # Scales also if mx, mn specified? Same notes and complaints as for the test
    # above.
    str_io.seek(0)
    array_to_file(arr, str_io, np.int8, mn=info['min'], mx=info['max'],
                  intercept=np.float32(2**120), nan2zero=False)
    data_back = array_from_file(arr.shape, np.int8, str_io)
    assert_array_equal(data_back, [-128, -128, 127])
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


def test_a2f_int_scaling():
    # Check that we can use integers for intercept and divslope
    arr = np.array([0, 1, 128, 255], dtype=np.uint8)
    fobj = BytesIO()
    back_arr = write_return(arr, fobj, np.uint8, intercept=1)
    assert_array_equal(back_arr, np.clip(arr - 1., 0, 255))
    back_arr = write_return(arr, fobj, np.uint8, divslope=2)
    assert_array_equal(back_arr, np.round(np.clip(arr / 2., 0, 255)))
    back_arr = write_return(arr, fobj, np.uint8, intercept=1, divslope=2)
    assert_array_equal(back_arr, np.round(np.clip((arr - 1.) / 2., 0, 255)))
    back_arr = write_return(arr, fobj, np.int16, intercept=1, divslope=2)
    assert_array_equal(back_arr, np.round((arr - 1.) / 2.))


def test_a2f_scaled_unscaled():
    # Test behavior of array_to_file when writing different types with and
    # without scaling
    fobj = BytesIO()
    for in_dtype, out_dtype, intercept, divslope in itertools.product(
        NUMERIC_TYPES,
        NUMERIC_TYPES,
        (0, 0.5, -1, 1),
        (1, 0.5, 2)):
        mn_in, mx_in = _dt_min_max(in_dtype)
        nan_val = np.nan if in_dtype in CFLOAT_TYPES else 10
        arr = np.array([mn_in, -1, 0, 1, mx_in, nan_val], dtype=in_dtype)
        mn_out, mx_out = _dt_min_max(out_dtype)
        nan_fill = -intercept / divslope
        if out_dtype in IUINT_TYPES:
            nan_fill = np.round(nan_fill)
        if (in_dtype in CFLOAT_TYPES and not mn_out <= nan_fill <= mx_out):
            assert_raises(ValueError,
                          array_to_file,
                          arr,
                          fobj,
                          out_dtype=out_dtype,
                          divslope=divslope,
                          intercept=intercept)
            continue
        back_arr = write_return(arr, fobj,
                                out_dtype=out_dtype,
                                divslope=divslope,
                                intercept=intercept)
        exp_back = arr.copy()
        if out_dtype in IUINT_TYPES:
            exp_back[np.isnan(exp_back)] = 0
        if in_dtype not in COMPLEX_TYPES:
            exp_back = exp_back.astype(float)
        if intercept != 0:
            exp_back -= intercept
        if divslope != 1:
            exp_back /= divslope
        if out_dtype in IUINT_TYPES:
            exp_back = np.round(exp_back).astype(float)
            exp_back = np.clip(exp_back, *shared_range(float, out_dtype))
            exp_back = exp_back.astype(out_dtype)
        else:
            exp_back = exp_back.astype(out_dtype)
        # Allow for small differences in large numbers
        assert_allclose_safely(back_arr, exp_back)


def test_a2f_nanpos():
    # Strange behavior of nan2zero
    arr = np.array([np.nan])
    fobj = BytesIO()
    back_arr = write_return(arr, fobj, np.int8, divslope=2)
    assert_array_equal(back_arr, 0)
    back_arr = write_return(arr, fobj, np.int8, intercept=10, divslope=2)
    assert_array_equal(back_arr, -5)


def test_a2f_bad_scaling():
    # Test that pathological scalers raise an error
    NUMERICAL_TYPES = sum([np.sctypes[key] for key in ['int',
                                                       'uint',
                                                       'float',
                                                       'complex']],
                         [])
    for in_type, out_type, slope, inter in itertools.product(
        NUMERICAL_TYPES,
        NUMERICAL_TYPES,
        (None, 1, 0, np.nan, -np.inf, np.inf),
        (0, np.nan, -np.inf, np.inf)):
        arr = np.ones((2,), dtype=in_type)
        fobj = BytesIO()
        if (slope, inter) == (1, 0):
            assert_array_equal(arr,
                               write_return(arr, fobj, out_type,
                                            intercept=inter,
                                            divslope=slope))
        elif (slope, inter) == (None, 0):
            assert_array_equal(0,
                               write_return(arr, fobj, out_type,
                                            intercept=inter,
                                            divslope=slope))
        else:
            assert_raises(ValueError,
                          array_to_file,
                          arr,
                          fobj,
                          np.int8,
                          intercept=inter,
                          divslope=slope)


def test_a2f_nan2zero_range():
    # array_to_file should check if nan can be represented as zero
    # This comes about when the writer can't write the value (-intercept /
    # divslope) because it does not fit in the output range.  Input clipping
    # should not affect this
    fobj = BytesIO()
    # No problem for input integer types - they don't have NaNs
    for dt in INT_TYPES:
        arr_no_nan = np.array([-1, 0, 1, 2], dtype=dt)
        # No errors from explicit thresholding (nor for input float types)
        back_arr = write_return(arr_no_nan, fobj, np.int8, mn=1, nan2zero=True)
        assert_array_equal([1, 1, 1, 2], back_arr)
        back_arr = write_return(arr_no_nan, fobj, np.int8, mx=-1, nan2zero=True)
        assert_array_equal([-1, -1, -1, -1], back_arr)
        # Pushing zero outside the output data range does not generate error
        back_arr = write_return(arr_no_nan, fobj, np.int8, intercept=129, nan2zero=True)
        assert_array_equal([-128, -128, -128, -127], back_arr)
        back_arr = write_return(arr_no_nan, fobj, np.int8,
                                intercept=257.1, divslope=2, nan2zero=True)
        assert_array_equal([-128, -128, -128, -128], back_arr)
    for dt in CFLOAT_TYPES:
        arr = np.array([-1, 0, 1, np.nan], dtype=dt)
        # Error occurs for arrays without nans too
        arr_no_nan = np.array([-1, 0, 1, 2], dtype=dt)
        # No errors from explicit thresholding
        # mn thresholding excluding zero
        assert_array_equal([1, 1, 1, 0],
                           write_return(arr, fobj, np.int8, mn=1))
        # mx thresholding excluding zero
        assert_array_equal([-1, -1, -1, 0],
                           write_return(arr, fobj, np.int8, mx=-1))
        # Errors from datatype threshold after scaling
        back_arr = write_return(arr, fobj, np.int8, intercept=128)
        assert_array_equal([-128, -128, -127, -128], back_arr)
        assert_raises(ValueError, write_return, arr, fobj, np.int8, intercept=129)
        assert_raises(ValueError, write_return, arr_no_nan, fobj, np.int8, intercept=129)
        # OK with nan2zero false, but we get whatever nan casts to
        nan_cast = np.array(np.nan).astype(np.int8)
        back_arr = write_return(arr, fobj, np.int8, intercept=129, nan2zero=False)
        assert_array_equal([-128, -128, -128, nan_cast], back_arr)
        # divslope
        back_arr = write_return(arr, fobj, np.int8, intercept=256, divslope=2)
        assert_array_equal([-128, -128, -128, -128], back_arr)
        assert_raises(ValueError, write_return, arr, fobj, np.int8,
                      intercept=257.1, divslope=2)
        assert_raises(ValueError, write_return, arr_no_nan, fobj, np.int8,
                      intercept=257.1, divslope=2)
        # OK with nan2zero false
        back_arr = write_return(arr, fobj, np.int8,
                                intercept=257.1, divslope=2, nan2zero=False)
        assert_array_equal([-128, -128, -128, nan_cast], back_arr)


def test_a2f_non_numeric():
    # Reminder that we may get structured dtypes
    dt = np.dtype([('f1', 'f'), ('f2', 'i2')])
    arr = np.zeros((2,), dtype=dt)
    arr['f1'] = 0.4, 0.6
    arr['f2'] = 10, 12
    fobj = BytesIO()
    back_arr = write_return(arr, fobj, dt)
    assert_array_equal(back_arr, arr)
    # Some versions of numpy can cast structured types to float, others not
    try:
        arr.astype(float)
    except ValueError:
        pass
    else:
        back_arr = write_return(arr, fobj, float)
        assert_array_equal(back_arr, arr.astype(float))
    # mn, mx never work for structured types
    assert_raises(ValueError, write_return, arr, fobj, float, mn=0)
    assert_raises(ValueError, write_return, arr, fobj, float, mx=10)


def write_return(data, fileobj, out_dtype, *args, **kwargs):
    fileobj.truncate(0)
    fileobj.seek(0)
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
    # Non-zero intercept still generates floats
    assert_dt_equal(apply_read_scaling(i16_arr, 1.0, 1.0).dtype, float)
    assert_dt_equal(apply_read_scaling(
        np.zeros((1,), dtype=np.int32), 1.0, 1.0).dtype, float)
    assert_dt_equal(apply_read_scaling(
        np.zeros((1,), dtype=np.int64), 1.0, 1.0).dtype, float)


def test_apply_read_scaling_ints():
    # Test that apply_read_scaling copes with integer scaling inputs
    arr = np.arange(10, dtype=np.int16)
    assert_array_equal(apply_read_scaling(arr, 1, 0), arr)
    assert_array_equal(apply_read_scaling(arr, 1, 1), arr + 1)
    assert_array_equal(apply_read_scaling(arr, 2, 1), arr * 2 + 1)


def test_apply_read_scaling_nones():
    # Check that we can pass None as slope and inter to apply read scaling
    arr = np.arange(10, dtype=np.int16)
    assert_array_equal(apply_read_scaling(arr, None, None), arr)
    assert_array_equal(apply_read_scaling(arr, 2, None), arr * 2)
    assert_array_equal(apply_read_scaling(arr, None, 1), arr + 1)


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
    assert_equal(bio.getvalue(), b'\x00'*10000)
    bio.seek(0)
    bio.truncate(0)
    write_zeros(bio, 10000, 256)
    assert_equal(bio.getvalue(), b'\x00'*10000)
    bio.seek(0)
    bio.truncate(0)
    write_zeros(bio, 200, 256)
    assert_equal(bio.getvalue(), b'\x00'*200)


def test_seek_tell():
    # Test seek tell routine
    bio = BytesIO()
    in_files = bio, 'test.bin', 'test.gz', 'test.bz2'
    start = 10
    end = 100
    diff = end - start
    tail = 7
    with InTemporaryDirectory():
        for in_file, write0 in itertools.product(in_files, (False, True)):
            st = functools.partial(seek_tell, write0=write0)
            bio.seek(0)
            # First write the file
            with BinOpener(in_file, 'wb') as fobj:
                assert_equal(fobj.tell(), 0)
                # already at position - OK
                st(fobj, 0)
                assert_equal(fobj.tell(), 0)
                # Move position by writing
                fobj.write(b'\x01' * start)
                assert_equal(fobj.tell(), start)
                # Files other than BZ2Files can seek forward on write, leaving
                # zeros in their wake.  BZ2Files can't seek when writing, unless
                # we enable the write0 flag to seek_tell
                if not write0 and in_file == 'test.bz2': # Can't seek write in bz2
                    # write the zeros by hand for the read test below
                    fobj.write(b'\x00' * diff)
                else:
                    st(fobj, end)
                    assert_equal(fobj.tell(), end)
                # Write tail
                fobj.write(b'\x02' * tail)
            bio.seek(0)
            # Now read back the file testing seek_tell in reading mode
            with BinOpener(in_file, 'rb') as fobj:
                assert_equal(fobj.tell(), 0)
                st(fobj, 0)
                assert_equal(fobj.tell(), 0)
                st(fobj, start)
                assert_equal(fobj.tell(), start)
                st(fobj, end)
                assert_equal(fobj.tell(), end)
                # Seek anywhere works in read mode for all files
                st(fobj, 0)
            bio.seek(0)
            # Check we have the expected written output
            with BinOpener(in_file, 'rb') as fobj:
                assert_equal(fobj.read(),
                             b'\x01' * start + b'\x00' * diff + b'\x02' * tail)
        for in_file in ('test2.gz', 'test2.bz2'):
            # Check failure of write seek backwards
            with BinOpener(in_file, 'wb') as fobj:
                fobj.write(b'g' * 10)
                assert_equal(fobj.tell(), 10)
                seek_tell(fobj, 10)
                assert_equal(fobj.tell(), 10)
                assert_raises(IOError, seek_tell, fobj, 5)
            # Make sure read seeks don't affect file
            with BinOpener(in_file, 'rb') as fobj:
                seek_tell(fobj, 10)
                seek_tell(fobj, 0)
            with BinOpener(in_file, 'rb') as fobj:
                assert_equal(fobj.read(), b'g' * 10)


def test_seek_tell_logic():
    # Test logic of seek_tell write0 with dummy class
    # Seek works? OK
    bio = BytesIO()
    seek_tell(bio, 10)
    assert_equal(bio.tell(), 10)
    class BabyBio(BytesIO):
        def seek(self, *args):
            raise IOError()
    bio = BabyBio()
    # Fresh fileobj, position 0, can't seek - error
    assert_raises(IOError, bio.seek, 10)
    # Put fileobj in correct position by writing
    ZEROB = b'\x00'
    bio.write(ZEROB * 10)
    seek_tell(bio, 10) # already there, nothing to do
    assert_equal(bio.tell(), 10)
    assert_equal(bio.getvalue(), ZEROB * 10)
    # Try write zeros to get to new position
    assert_raises(IOError, bio.seek, 20)
    seek_tell(bio, 20, write0=True)
    assert_equal(bio.getvalue(), ZEROB * 20)


def test_BinOpener():
    # Test that BinOpener does add '.mgz' as gzipped file type
    with InTemporaryDirectory():
        with BinOpener('test.gz', 'w') as fobj:
            assert_true(hasattr(fobj.fobj, 'compress'))
        with BinOpener('test.mgz', 'w') as fobj:
            assert_true(hasattr(fobj.fobj, 'compress'))


def test_fname_ext_ul_case():
    # Get filename ignoring the case of the filename extension
    with InTemporaryDirectory():
        with open('afile.TXT', 'wt') as fobj:
            fobj.write('Interesting information')
        # OSX usually has case-insensitive file systems; Windows also
        os_cares_case = not exists('afile.txt')
        with open('bfile.txt', 'wt') as fobj:
            fobj.write('More interesting information')
        # If there is no file, the case doesn't change
        assert_equal(fname_ext_ul_case('nofile.txt'), 'nofile.txt')
        assert_equal(fname_ext_ul_case('nofile.TXT'), 'nofile.TXT')
        # If there is a file, accept upper or lower case for ext
        if os_cares_case:
            assert_equal(fname_ext_ul_case('afile.txt'), 'afile.TXT')
            assert_equal(fname_ext_ul_case('bfile.TXT'), 'bfile.txt')
        else:
            assert_equal(fname_ext_ul_case('afile.txt'), 'afile.txt')
            assert_equal(fname_ext_ul_case('bfile.TXT'), 'bfile.TXT')
        assert_equal(fname_ext_ul_case('afile.TXT'), 'afile.TXT')
        assert_equal(fname_ext_ul_case('bfile.txt'), 'bfile.txt')
        # Not mixed case though
        assert_equal(fname_ext_ul_case('afile.TxT'), 'afile.TxT')


def test_allopen():
    # This import into volumeutils is for compatibility.  The code is the
    # ``openers`` module.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
        msg = b'tiddle pom'
        sobj = BytesIO(msg)
        fobj = allopen(sobj)
        assert_equal(fobj.read(), msg)
        # mode is gently ignored
        fobj = allopen(sobj, mode='r')


def test_allopen_compresslevel():
    # We can set the default compression level with the module global
    # Get some data to compress
    with open(__file__, 'rb') as fobj:
        my_self = fobj.read()
    # Prepare loop
    fname = 'test.gz'
    sizes = {}
    # Stash module global
    from .. import volumeutils as vu
    original_compress_level = vu.default_compresslevel
    assert_equal(original_compress_level, 1)
    try:
        with InTemporaryDirectory():
            for compresslevel in ('default', 1, 9):
                if compresslevel != 'default':
                    vu.default_compresslevel = compresslevel
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with allopen(fname, 'wb') as fobj:
                        fobj.write(my_self)
                with open(fname, 'rb') as fobj:
                    my_selves_smaller = fobj.read()
                sizes[compresslevel] = len(my_selves_smaller)
            assert_equal(sizes['default'], sizes[1])
            assert_true(sizes[1] > sizes[9])
    finally:
        vu.default_compresslevel = original_compress_level


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
    assert_equal(d, {'x': 0, 's': b''})


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


def test__write_data():
    # Test private utility function for writing data
    itp = itertools.product

    def assert_rt(data,
                  shape,
                  out_dtype,
                  order='F',
                  in_cast = None,
                  pre_clips = None,
                  inter = 0.,
                  slope = 1.,
                  post_clips = None,
                  nan_fill = None):
        sio = BytesIO()
        to_write = data.reshape(shape)
        # to check that we didn't modify in-place
        backup = to_write.copy()
        nan_positions = np.isnan(to_write)
        have_nans = np.any(nan_positions)
        if have_nans and nan_fill is None and not out_dtype.type == 'f':
            raise ValueError("Cannot handle this case")
        _write_data(to_write, sio, out_dtype, order, in_cast, pre_clips, inter,
                    slope, post_clips, nan_fill)
        arr = np.ndarray(shape, out_dtype, buffer=sio.getvalue(),
                         order=order)
        expected = to_write.copy()
        if have_nans and not nan_fill is None:
            expected[nan_positions] = nan_fill * slope + inter
        assert_array_equal(arr * slope + inter, expected)
        assert_array_equal(to_write, backup)

    # check shape writing
    for shape, order in itp(
        ((24,), (24, 1), (24, 1, 1), (1, 24), (1, 1, 24), (2, 3, 4),
         (6, 1, 4), (1, 6, 4), (6, 4, 1)),
        'FC'):
        assert_rt(np.arange(24), shape, np.int16, order=order)

    # check defense against modifying data in-place
    for in_cast, pre_clips, inter, slope, post_clips, nan_fill in itp(
        (None, np.float32),
        (None, (-1, 25)),
        (0., 1.),
        (1., 0.5),
        (None, (-2, 49)),
        (None, 1)):
        data = np.arange(24).astype(np.float32)
        assert_rt(data, shape, np.int16,
                  in_cast = in_cast,
                  pre_clips = pre_clips,
                  inter = inter,
                  slope = slope,
                  post_clips = post_clips,
                  nan_fill = nan_fill)
        # Check defense against in-place modification with nans present
        if not nan_fill is None:
            data[1] = np.nan
            assert_rt(data, shape, np.int16,
                      in_cast = in_cast,
                      pre_clips = pre_clips,
                      inter = inter,
                      slope = slope,
                      post_clips = post_clips,
                      nan_fill = nan_fill)
