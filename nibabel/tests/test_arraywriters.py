""" Testing array writer objects

Array writers have init signature::

    def __init__(self, array, out_dtype=None, order='F')

and methods

* to_fileobj(fileobj, offset=None)

They do have attributes:

* array
* out_dtype
* order

They may have attributes:

* slope
* inter

They are designed to write arrays to a fileobj with reasonable memory
efficiency.

Subclasses of array writers may be able to scale the array or apply an
intercept, or do something else to make sense of conversions between float and
int, or between larger ints and smaller.
"""

from platform import python_compiler, machine

import numpy as np

from ..py3k import BytesIO

from ..arraywriters import (SlopeInterArrayWriter, SlopeArrayWriter,
                            WriterError, ScalingError, ArrayWriter,
                            make_array_writer, get_slope_inter)

from ..casting import int_abs, type_info

from ..volumeutils import array_from_file, apply_read_scaling

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false,
                        assert_equal, assert_not_equal,
                        assert_raises)


FLOAT_TYPES = np.sctypes['float']
COMPLEX_TYPES = np.sctypes['complex']
INT_TYPES = np.sctypes['int']
UINT_TYPES = np.sctypes['uint']
CFLOAT_TYPES = FLOAT_TYPES + COMPLEX_TYPES
IUINT_TYPES = INT_TYPES + UINT_TYPES
NUMERIC_TYPES = CFLOAT_TYPES + IUINT_TYPES


def round_trip(writer, order='F', nan2zero=True, apply_scale=True):
    sio = BytesIO()
    arr = writer.array
    writer.to_fileobj(sio, order, nan2zero=nan2zero)
    data_back = array_from_file(arr.shape, writer.out_dtype, sio, order=order)
    slope, inter = get_slope_inter(writer)
    if apply_scale:
        data_back = apply_read_scaling(data_back, slope, inter)
    return data_back


def test_arraywriters():
    # Test initialize
    # Simple cases
    if machine() == 'sparc64' and python_compiler().startswith('GCC'):
        # bus errors on at least np 1.4.1 through 1.6.1 for complex
        test_types = FLOAT_TYPES + IUINT_TYPES
    else:
        test_types = NUMERIC_TYPES
    for klass in (SlopeInterArrayWriter, SlopeArrayWriter, ArrayWriter):
        for type in test_types:
            arr = np.arange(10, dtype=type)
            aw = klass(arr)
            assert_true(aw.array is arr)
            assert_equal(aw.out_dtype, arr.dtype)
            assert_array_equal(arr, round_trip(aw))
            # Byteswapped is OK
            bs_arr = arr.byteswap().newbyteorder('S')
            bs_aw = klass(bs_arr)
            # assert against original array because POWER7 was running into
            # trouble using the byteswapped array (bs_arr)
            assert_array_equal(arr, round_trip(bs_aw))
            bs_aw2 = klass(bs_arr, arr.dtype)
            assert_array_equal(arr, round_trip(bs_aw2))
            # 2D array
            arr2 = np.reshape(arr, (2, 5))
            a2w = klass(arr2)
            # Default out - in order is Fortran
            arr_back = round_trip(a2w)
            assert_array_equal(arr2, arr_back)
            arr_back = round_trip(a2w, 'F')
            assert_array_equal(arr2, arr_back)
            # C order works as well
            arr_back = round_trip(a2w, 'C')
            assert_array_equal(arr2, arr_back)
            assert_true(arr_back.flags.c_contiguous)


def test_scaling_needed():
    # Structured types return True if dtypes same, raise error otherwise
    dt_def = [('f', 'i4')]
    arr = np.ones(10, dt_def)
    for t in NUMERIC_TYPES:
        assert_raises(WriterError, ArrayWriter, arr, t)
        narr = np.ones(10, t)
        assert_raises(WriterError, ArrayWriter, narr, dt_def)
    assert_false(ArrayWriter(arr).scaling_needed())
    assert_false(ArrayWriter(arr, dt_def).scaling_needed())
    # Any numeric type that can cast, needs no scaling
    for in_t in NUMERIC_TYPES:
        for out_t in NUMERIC_TYPES:
            if np.can_cast(in_t, out_t):
                aw = ArrayWriter(np.ones(10, in_t), out_t)
                assert_false(aw.scaling_needed())
    for in_t in NUMERIC_TYPES:
        # Numeric types to complex never need scaling
        arr = np.ones(10, in_t)
        for out_t in COMPLEX_TYPES:
            assert_false(ArrayWriter(arr, out_t).scaling_needed())
    # Attempts to scale from complex to anything else fails
    for in_t in COMPLEX_TYPES:
        for out_t in FLOAT_TYPES + IUINT_TYPES:
            arr = np.ones(10, in_t)
            assert_raises(WriterError, ArrayWriter, arr, out_t)
    # Scaling from anything but complex to floats is OK
    for in_t in FLOAT_TYPES + IUINT_TYPES:
        arr = np.ones(10, in_t)
        for out_t in FLOAT_TYPES:
            assert_false(ArrayWriter(arr, out_t).scaling_needed())
    # For any other output type, arrays with no data don't need scaling
    for in_t in FLOAT_TYPES + IUINT_TYPES:
        arr_0 = np.zeros(10, in_t)
        arr_e = []
        for out_t in IUINT_TYPES:
            assert_false(ArrayWriter(arr_0, out_t).scaling_needed())
            assert_false(ArrayWriter(arr_e, out_t).scaling_needed())
    # Going to (u)ints, non-finite arrays don't need scaling
    for in_t in FLOAT_TYPES:
        arr_nan = np.zeros(10, in_t) + np.nan
        arr_inf = np.zeros(10, in_t) + np.inf
        arr_minf = np.zeros(10, in_t) - np.inf
        arr_mix = np.array([np.nan, np.inf, -np.inf], dtype=in_t)
        for out_t in IUINT_TYPES:
            for arr in (arr_nan, arr_inf, arr_minf, arr_mix):
                assert_false(ArrayWriter(arr, out_t).scaling_needed())
    # Floats as input always need scaling
    for in_t in FLOAT_TYPES:
        arr = np.ones(10, in_t)
        for out_t in IUINT_TYPES:
            # We need an arraywriter that will tolerate construction when
            # scaling is needed
            assert_true(SlopeArrayWriter(arr, out_t).scaling_needed())
    # in-range (u)ints don't need scaling
    for in_t in IUINT_TYPES:
        in_info = np.iinfo(in_t)
        in_min, in_max = in_info.min, in_info.max
        for out_t in IUINT_TYPES:
            out_info = np.iinfo(out_t)
            out_min, out_max = out_info.min, out_info.max
            if in_min >= out_min and in_max <= out_max:
                arr = np.array([in_min, in_max], in_t)
                assert_true(np.can_cast(arr.dtype, out_t))
                # We've already tested this with can_cast above, but...
                assert_false(ArrayWriter(arr, out_t).scaling_needed())
                continue
            # The output data type does not include the input data range
            max_min = max(in_min, out_min) # 0 for input or output uint
            min_max = min(in_max, out_max)
            arr = np.array([max_min, min_max], in_t)
            assert_false(ArrayWriter(arr, out_t).scaling_needed())
            assert_true(SlopeInterArrayWriter(arr + 1, out_t).scaling_needed())
            if in_t in INT_TYPES:
                assert_true(SlopeInterArrayWriter(arr - 1, out_t).scaling_needed())


def test_special_rt():
    # Test that zeros; none finite - round trip to zeros
    for arr in (np.array([np.inf, np.nan, -np.inf]),
                np.zeros((3,))):
        for in_dtt in FLOAT_TYPES:
            for out_dtt in IUINT_TYPES:
                for klass in (ArrayWriter, SlopeArrayWriter,
                              SlopeInterArrayWriter):
                    aw = klass(arr.astype(in_dtt), out_dtt)
                    assert_equal(get_slope_inter(aw), (1, 0))
                    assert_array_equal(round_trip(aw), 0)


def test_high_int2uint():
    # Need to take care of high values when testing whether values are already
    # in range.  There was a bug here were the comparison was in floating point,
    # and therefore not exact, and 2**63 appeared to be in range for np.int64
    arr = np.array([2**63], dtype=np.uint64)
    out_type = np.int64
    aw = SlopeInterArrayWriter(arr, out_type)
    assert_equal(aw.inter, 2**63)


def test_slope_inter_castable():
    # Test scaling for arraywriter instances
    # Test special case of all zeros
    for in_dtt in FLOAT_TYPES + IUINT_TYPES:
        for out_dtt in NUMERIC_TYPES:
            for klass in (ArrayWriter, SlopeArrayWriter, SlopeInterArrayWriter):
                arr = np.zeros((5,), dtype=in_dtt)
                aw = klass(arr, out_dtt) # no error
    # Test special case of none finite
    arr = np.array([np.inf, np.nan, -np.inf])
    for in_dtt in FLOAT_TYPES:
        for out_dtt in FLOAT_TYPES + IUINT_TYPES:
            for klass in (ArrayWriter, SlopeArrayWriter, SlopeInterArrayWriter):
                aw = klass(arr.astype(in_dtt), out_dtt) # no error
    for in_dtt, out_dtt, arr, slope_only, slope_inter, neither in (
        (np.float32, np.float32, 1, True, True, True),
        (np.float64, np.float32, 1, True, True, True),
        (np.float32, np.complex128, 1, True, True, True),
        (np.uint32, np.complex128, 1, True, True, True),
        (np.int64, np.float32, 1, True, True, True),
        (np.float32, np.int16, 1, True, True, False),
        (np.complex128, np.float32, 1, False, False, False),
        (np.complex128, np.int16, 1, False, False, False),
        (np.uint8, np.int16, 1, True, True, True),
        # The following tests depend on the input data
        (np.uint16, np.int16, 1, True, True, True), # 1 is in range
        (np.uint16, np.int16, 2**16-1, True, True, False), # This not in range
        (np.uint16, np.int16, (0, 2**16-1), True, True, False),
        (np.uint16, np.uint8, 1, True, True, True),
        (np.int16, np.uint16, 1, True, True, True), # in range
        (np.int16, np.uint16, -1, True, True, False), # flip works for scaling
        (np.int16, np.uint16, (-1, 1), False, True, False), # not with +-
        (np.int8, np.uint16, 1, True, True, True), # in range
        (np.int8, np.uint16, -1, True, True, False), # flip works for scaling
        (np.int8, np.uint16, (-1, 1), False, True, False), # not with +-
    ):
        # data for casting
        data = np.array(arr, dtype=in_dtt)
        # With scaling but no intercept
        if slope_only:
            aw = SlopeArrayWriter(data, out_dtt)
        else:
            assert_raises(WriterError, SlopeArrayWriter, data, out_dtt)
        # With scaling and intercept
        if slope_inter:
            aw = SlopeInterArrayWriter(data, out_dtt)
        else:
            assert_raises(WriterError, SlopeInterArrayWriter, data, out_dtt)
        # With neither
        if neither:
            aw = ArrayWriter(data, out_dtt)
        else:
            assert_raises(WriterError, ArrayWriter, data, out_dtt)


def test_calculate_scale():
    # Test for special cases in scale calculation
    npa = np.array
    SIAW = SlopeInterArrayWriter
    SAW = SlopeArrayWriter
    # Offset handles scaling when it can
    aw = SIAW(npa([-2, -1], dtype=np.int8), np.uint8)
    assert_equal(get_slope_inter(aw), (1.0, -2.0))
    # Sign flip handles these cases
    aw = SAW(npa([-2, -1], dtype=np.int8), np.uint8)
    assert_equal(get_slope_inter(aw), (-1.0, 0.0))
    aw = SAW(npa([-2, 0], dtype=np.int8), np.uint8)
    assert_equal(get_slope_inter(aw), (-1.0, 0.0))
    # But not when min magnitude is too large (scaling mechanism kicks in)
    aw = SAW(npa([-510, 0], dtype=np.int16), np.uint8)
    assert_equal(get_slope_inter(aw), (-2.0, 0.0))
    # Or for floats (attempts to expand across range)
    aw = SAW(npa([-2, 0], dtype=np.float32), np.uint8)
    assert_not_equal(get_slope_inter(aw), (-1.0, 0.0))
    # Case where offset handles scaling
    aw = SIAW(npa([-1, 1], dtype=np.int8), np.uint8)
    assert_equal(get_slope_inter(aw), (1.0, -1.0))
    # Can't work for no offset case
    assert_raises(WriterError, SAW, npa([-1, 1], dtype=np.int8), np.uint8)
    # Offset trick can't work when max is out of range
    aw = SIAW(npa([-1, 255], dtype=np.int16), np.uint8)
    slope_inter = get_slope_inter(aw)
    assert_not_equal(slope_inter, (1.0, -1.0))


def test_resets():
    # Test reset of values, caching of scales
    for klass, inp, outp in ((SlopeInterArrayWriter, (1, 511), (2.0, 1.0)),
                             (SlopeArrayWriter, (0, 510), (2.0, 0.0))):
        arr = np.array(inp)
        outp = np.array(outp)
        aw = klass(arr, np.uint8)
        assert_array_equal(get_slope_inter(aw), outp)
        aw.calc_scale() # cached no change
        assert_array_equal(get_slope_inter(aw), outp)
        aw.calc_scale(force=True) # same data, no change
        assert_array_equal(get_slope_inter(aw), outp)
        # Change underlying array
        aw.array[:] = aw.array * 2
        aw.calc_scale() # cached still
        assert_array_equal(get_slope_inter(aw), outp)
        aw.calc_scale(force=True) # new data, change
        assert_array_equal(get_slope_inter(aw), outp * 2)
        # Test reset
        aw.reset()
        assert_array_equal(get_slope_inter(aw), (1.0, 0.0))


def test_no_offset_scale():
    # Specific tests of no-offset scaling
    SAW = SlopeArrayWriter
    # Floating point
    for data in ((-128, 127),
                  (-128, 126),
                  (-128, -127),
                  (-128, 0),
                  (-128, -1),
                  (126, 127),
                  (-127, 127)):
        aw = SAW(np.array(data, dtype=np.float32), np.int8)
        assert_equal(aw.slope, 1.0)
    aw = SAW(np.array([-126, 127 * 2.0], dtype=np.float32), np.int8)
    assert_equal(aw.slope, 2)
    aw = SAW(np.array([-128 * 2.0, 127], dtype=np.float32), np.int8)
    assert_equal(aw.slope, 2)
    # Test that nasty abs behavior does not upset us
    n = -2**15
    aw = SAW(np.array([n, n], dtype=np.int16), np.uint8)
    assert_array_almost_equal(aw.slope, n / 255.0, 5)


def test_with_offset_scale():
    # Tests of specific cases in slope, inter
    SIAW = SlopeInterArrayWriter
    aw = SIAW(np.array([0, 127], dtype=np.int8), np.uint8)
    assert_equal((aw.slope, aw.inter), (1, 0)) # in range
    aw = SIAW(np.array([-1, 126], dtype=np.int8), np.uint8)
    assert_equal((aw.slope, aw.inter), (1, -1)) # offset only
    aw = SIAW(np.array([-1, 254], dtype=np.int16), np.uint8)
    assert_equal((aw.slope, aw.inter), (1, -1)) # offset only
    aw = SIAW(np.array([-1, 255], dtype=np.int16), np.uint8)
    assert_not_equal((aw.slope, aw.inter), (1, -1)) # Too big for offset only
    aw = SIAW(np.array([-256, -2], dtype=np.int16), np.uint8)
    assert_equal((aw.slope, aw.inter), (1, -256)) # offset only
    aw = SIAW(np.array([-256, -2], dtype=np.int16), np.int8)
    assert_equal((aw.slope, aw.inter), (1, -129)) # offset only


def test_io_scaling():
    # Test scaling works for max, min when going from larger to smaller type,
    # and from float to integer.
    bio = BytesIO()
    for in_type, out_type, err in ((np.int16, np.int16, None),
                                   (np.int16, np.int8, None),
                                   (np.uint16, np.uint8, None),
                                   (np.int32, np.int8, None),
                                   (np.float32, np.uint8, None),
                                   (np.float32, np.int16, None)):
        out_dtype = np.dtype(out_type)
        arr = np.zeros((3,), dtype=in_type)
        info = type_info(in_type)
        arr[0], arr[1] = info['min'], info['max']
        aw = SlopeInterArrayWriter(arr, out_dtype, calc_scale=False)
        if not err is None:
            assert_raises(err, aw.calc_scale)
            continue
        aw.calc_scale()
        aw.to_fileobj(bio)
        bio.seek(0)
        arr2 = array_from_file(arr.shape, out_dtype, bio)
        arr3 = apply_read_scaling(arr2, aw.slope, aw.inter)
        # Max rounding error for integer type
        max_miss = aw.slope / 2.
        assert_true(np.all(np.abs(arr - arr3) <= max_miss))
        bio.truncate(0)
        bio.seek(0)


def test_nan2zero():
    # Test conditions under which nans written to zero
    arr = np.array([np.nan, 99.], dtype=np.float32)
    aw = SlopeInterArrayWriter(arr, np.float32)
    data_back = round_trip(aw)
    assert_array_equal(np.isnan(data_back), [True, False])
    # nan2zero ignored for floats
    data_back = round_trip(aw, nan2zero=True)
    assert_array_equal(np.isnan(data_back), [True, False])
    # Integer output with nan2zero gives zero
    aw = SlopeInterArrayWriter(arr, np.int32)
    data_back = round_trip(aw, nan2zero=True)
    assert_array_equal(data_back, [0, 99])
    # Integer output with nan2zero=False gives whatever astype gives
    data_back = round_trip(aw, nan2zero=False)
    astype_res = np.array(np.nan).astype(np.int32) * aw.slope + aw.inter
    assert_array_equal(data_back, [astype_res, 99])


def test_byte_orders():
    arr = np.arange(10, dtype=np.int32)
    # Test endian read/write of types not requiring scaling
    for tp in (np.uint64, np.float, np.complex):
        dt = np.dtype(tp)
        for code in '<>':
            ndt = dt.newbyteorder(code)
            for klass in (SlopeInterArrayWriter, SlopeArrayWriter,
                          ArrayWriter):
                aw = klass(arr, ndt)
                data_back = round_trip(aw)
                assert_array_almost_equal(arr, data_back)


def test_writers_roundtrip():
    ndt = np.dtype(np.float)
    arr = np.arange(3, dtype=ndt)
    # intercept
    aw = SlopeInterArrayWriter(arr, ndt, calc_scale=False)
    aw.inter = 1.0
    data_back = round_trip(aw)
    assert_array_equal(data_back, arr)
    # scaling
    aw.slope = 2.0
    data_back = round_trip(aw)
    assert_array_equal(data_back, arr)
    # if there is no valid data, we get zeros
    aw = SlopeInterArrayWriter(arr + np.nan, np.int32)
    data_back = round_trip(aw)
    assert_array_equal(data_back, np.zeros(arr.shape))
    # infs generate ints at same value as max
    arr[0] = np.inf
    aw = SlopeInterArrayWriter(arr, np.int32)
    data_back = round_trip(aw)
    assert_array_almost_equal(data_back, [2, 1, 2])


def test_to_float():
    start, stop = 0, 100
    for in_type in NUMERIC_TYPES:
        step = 1 if in_type in IUINT_TYPES else 0.5
        info = type_info(in_type)
        mn, mx = info['min'], info['max']
        arr = np.arange(start, stop, step, dtype=in_type)
        arr[0] = mn
        arr[-1] = mx
        for out_type in CFLOAT_TYPES:
            out_info = type_info(out_type)
            for klass in (SlopeInterArrayWriter, SlopeArrayWriter,
                          ArrayWriter):
                if in_type in COMPLEX_TYPES and out_type in FLOAT_TYPES:
                    assert_raises(WriterError, klass, arr, out_type)
                    continue
                aw = klass(arr, out_type)
                assert_true(aw.array is arr)
                assert_equal(aw.out_dtype, out_type)
                arr_back = round_trip(aw)
                assert_array_equal(arr.astype(out_type), arr_back)
                # Check too-big values overflowed correctly
                out_min, out_max = out_info['min'], out_info['max']
                assert_true(np.all(arr_back[arr > out_max] == np.inf))
                assert_true(np.all(arr_back[arr < out_min] == -np.inf))


def test_dumber_writers():
    arr = np.arange(10, dtype=np.float64)
    aw = SlopeArrayWriter(arr)
    aw.slope = 2.0
    assert_equal(aw.slope, 2.0)
    assert_raises(AttributeError, getattr, aw, 'inter')
    aw = ArrayWriter(arr)
    assert_raises(AttributeError, getattr, aw, 'slope')
    assert_raises(AttributeError, getattr, aw, 'inter')
    # Attempt at scaling should raise error for dumb type
    assert_raises(WriterError, ArrayWriter, arr, np.int16)


def test_writer_maker():
    arr = np.arange(10, dtype=np.float64)
    aw = make_array_writer(arr, np.float64)
    assert_true(isinstance(aw, SlopeInterArrayWriter))
    aw = make_array_writer(arr, np.float64, True, True)
    assert_true(isinstance(aw, SlopeInterArrayWriter))
    aw = make_array_writer(arr, np.float64, True, False)
    assert_true(isinstance(aw, SlopeArrayWriter))
    aw = make_array_writer(arr, np.float64, False, False)
    assert_true(isinstance(aw, ArrayWriter))
    assert_raises(ValueError, make_array_writer, arr, np.float64, False)
    assert_raises(ValueError, make_array_writer, arr, np.float64, False, True)
    # Does calc_scale get run by default?
    aw = make_array_writer(arr, np.int16, calc_scale=False)
    assert_equal((aw.slope, aw.inter), (1, 0))
    aw.calc_scale()
    slope, inter = aw.slope, aw.inter
    assert_false((slope, inter) == (1, 0))
    # Should run by default
    aw = make_array_writer(arr, np.int16)
    assert_equal((aw.slope, aw.inter), (slope, inter))
    aw = make_array_writer(arr, np.int16, calc_scale=True)
    assert_equal((aw.slope, aw.inter), (slope, inter))


def test_float_int_min_max():
    # Conversion between float and int
    for in_dt in FLOAT_TYPES:
        finf = type_info(in_dt)
        arr = np.array([finf['min'], finf['max']], dtype=in_dt)
        # Bug in numpy 1.6.2 on PPC leading to infs - abort
        if not np.all(np.isfinite(arr)):
            print 'Hit PPC max -> inf bug; skip in_type %s' % in_dt
            continue
        for out_dt in IUINT_TYPES:
            try:
                aw = SlopeInterArrayWriter(arr, out_dt)
            except ScalingError:
                continue
            arr_back_sc = round_trip(aw)
            assert_true(np.allclose(arr, arr_back_sc))


def test_int_int_min_max():
    # Conversion between (u)int and (u)int
    eps = np.finfo(np.float64).eps
    rtol = 1e-6
    for in_dt in IUINT_TYPES:
        iinf = np.iinfo(in_dt)
        arr = np.array([iinf.min, iinf.max], dtype=in_dt)
        for out_dt in IUINT_TYPES:
            try:
                aw = SlopeInterArrayWriter(arr, out_dt)
            except ScalingError:
                continue
            arr_back_sc = round_trip(aw)
            # integer allclose
            adiff = int_abs(arr - arr_back_sc)
            rdiff = adiff / (arr + eps)
            assert_true(np.all(rdiff < rtol))


def test_int_int_slope():
    # Conversion between (u)int and (u)int for slopes only
    eps = np.finfo(np.float64).eps
    rtol = 1e-7
    for in_dt in IUINT_TYPES:
        iinf = np.iinfo(in_dt)
        for out_dt in IUINT_TYPES:
            kinds = np.dtype(in_dt).kind + np.dtype(out_dt).kind
            if kinds in ('ii', 'uu', 'ui'):
                arrs = (np.array([iinf.min, iinf.max], dtype=in_dt),)
            elif kinds == 'iu':
                arrs = (np.array([iinf.min, 0], dtype=in_dt),
                        np.array([0, iinf.max], dtype=in_dt))
            for arr in arrs:
                try:
                    aw = SlopeArrayWriter(arr, out_dt)
                except ScalingError:
                    continue
                assert_false(aw.slope == 0)
                arr_back_sc = round_trip(aw)
                # integer allclose
                adiff = int_abs(arr - arr_back_sc)
                rdiff = adiff / (arr + eps)
                assert_true(np.all(rdiff < rtol))


def test_float_int_spread():
    # Test rounding error for spread of values
    powers = np.arange(-10, 10, 0.5)
    arr = np.concatenate((-10**powers, 10**powers))
    for in_dt in (np.float32, np.float64):
        arr_t = arr.astype(in_dt)
        for out_dt in IUINT_TYPES:
            aw = SlopeInterArrayWriter(arr_t, out_dt)
            arr_back_sc = round_trip(aw)
            # Get estimate for error
            max_miss = rt_err_estimate(arr_t,
                                       arr_back_sc.dtype,
                                       aw.slope,
                                       aw.inter)
            # Simulate allclose test with large atol
            diff = np.abs(arr_t - arr_back_sc)
            rdiff = diff / np.abs(arr_t)
            assert_true(np.all((diff <= max_miss) | (rdiff <= 1e-5)))


def rt_err_estimate(arr_t, out_dtype, slope, inter):
    # Error attributable to rounding
    max_int_miss = slope / 2.
    # Estimate error attributable to floating point slope / inter;
    # Remove inter / slope, put in a float type to simulate the type
    # promotion for the multiplication, apply slope / inter
    flt_there = (arr_t - inter) / slope
    flt_back = flt_there.astype(out_dtype) * slope + inter
    max_flt_miss = np.abs(arr_t - flt_back).max()
    # Max error is sum of rounding and fp error
    return max_int_miss + max_flt_miss


def test_rt_bias():
    # Check for bias in round trip
    rng = np.random.RandomState(20111214)
    mu, std, count = 100, 10, 100
    arr = rng.normal(mu, std, size=(count,))
    eps = np.finfo(np.float32).eps
    for in_dt in (np.float32, np.float64):
        arr_t = arr.astype(in_dt)
        for out_dt in IUINT_TYPES:
            aw = SlopeInterArrayWriter(arr_t, out_dt)
            arr_back_sc = round_trip(aw)
            bias = np.mean(arr_t - arr_back_sc)
            # Get estimate for error
            max_miss = rt_err_estimate(arr_t,
                                       arr_back_sc.dtype,
                                       aw.slope,
                                       aw.inter)
            # Hokey use of max_miss as a std estimate
            bias_thresh = np.max([max_miss / np.sqrt(count), eps])
            assert_true(np.abs(bias) < bias_thresh)
