""" Test floating point deconstructions and floor methods
"""
import sys

PY2 = sys.version_info[0] < 3

import numpy as np

from ..casting import (floor_exact, ceil_exact, as_int, FloatingError,
                       int_to_float, floor_log2, type_info, _check_nmant,
                       _check_maxexp, ok_floats, on_powerpc, have_binary128,
                       longdouble_precision_improved)
from ..testing import suppress_warnings

from nose import SkipTest
from nose.tools import assert_equal, assert_raises, assert_true, assert_false

IEEE_floats = [np.float32, np.float64]
try:
    np.float16
except AttributeError:  # float16 not present in np < 1.6
    have_float16 = False
else:
    have_float16 = True
if have_float16:
    IEEE_floats.append(np.float16)

LD_INFO = type_info(np.longdouble)


def test_type_info():
    # Test routine to get min, max, nmant, nexp
    for dtt in np.sctypes['int'] + np.sctypes['uint']:
        info = np.iinfo(dtt)
        infod = type_info(dtt)
        assert_equal(dict(min=info.min, max=info.max,
                          nexp=None, nmant=None,
                          minexp=None, maxexp=None,
                          width=np.dtype(dtt).itemsize), infod)
        assert_equal(infod['min'].dtype.type, dtt)
        assert_equal(infod['max'].dtype.type, dtt)
    for dtt in IEEE_floats + [np.complex64, np.complex64]:
        info = np.finfo(dtt)
        infod = type_info(dtt)
        assert_equal(dict(min=info.min, max=info.max,
                          nexp=info.nexp, nmant=info.nmant,
                          minexp=info.minexp, maxexp=info.maxexp,
                          width=np.dtype(dtt).itemsize),
                     infod)
        assert_equal(infod['min'].dtype.type, dtt)
        assert_equal(infod['max'].dtype.type, dtt)
    # What is longdouble?
    info = np.finfo(np.longdouble)
    dbl_info = np.finfo(np.float64)
    infod = type_info(np.longdouble)
    width = np.dtype(np.longdouble).itemsize
    vals = (info.nmant, info.nexp, width)
    # Information for PPC head / tail doubles from:
    # https://developer.apple.com/library/mac/#documentation/Darwin/Reference/Manpages/man3/float.3.html
    if vals in ((52, 11, 8),  # longdouble is same as double
                (63, 15, 12), (63, 15, 16),  # intel 80 bit
                (112, 15, 16),  # real float128
                (106, 11, 16)):  # PPC head, tail doubles, expected values
        assert_equal(dict(min=info.min, max=info.max,
                          minexp=info.minexp, maxexp=info.maxexp,
                          nexp=info.nexp, nmant=info.nmant, width=width),
                     infod)
    elif vals == (1, 1, 16):  # bust info for PPC head / tail longdoubles
        assert_equal(dict(min=dbl_info.min, max=dbl_info.max,
                          minexp=-1022, maxexp=1024,
                          nexp=11, nmant=106, width=16),
                     infod)
    elif vals == (52, 15, 12):
        exp_res = type_info(np.float64)
        exp_res['width'] = width
        assert_equal(exp_res, infod)
    else:
        raise ValueError("Unexpected float type to test")


def test_nmant():
    for t in IEEE_floats:
        assert_equal(type_info(t)['nmant'], np.finfo(t).nmant)
    if (LD_INFO['nmant'], LD_INFO['nexp']) == (63, 15):
        assert_equal(type_info(np.longdouble)['nmant'], 63)


def test_check_nmant_nexp():
    # Routine for checking number of sigificand digits and exponent
    for t in IEEE_floats:
        nmant = np.finfo(t).nmant
        maxexp = np.finfo(t).maxexp
        assert_true(_check_nmant(t, nmant))
        assert_false(_check_nmant(t, nmant - 1))
        assert_false(_check_nmant(t, nmant + 1))
        with suppress_warnings():  # overflow
            assert_true(_check_maxexp(t, maxexp))
        assert_false(_check_maxexp(t, maxexp - 1))
        with suppress_warnings():
            assert_false(_check_maxexp(t, maxexp + 1))
    # Check against type_info
    for t in ok_floats():
        ti = type_info(t)
        if ti['nmant'] != 106:  # This check does not work for PPC double pair
            assert_true(_check_nmant(t, ti['nmant']))
        assert_true(_check_maxexp(t, ti['maxexp']))


def test_as_int():
    # Integer representation of number
    assert_equal(as_int(2.0), 2)
    assert_equal(as_int(-2.0), -2)
    assert_raises(FloatingError, as_int, 2.1)
    assert_raises(FloatingError, as_int, -2.1)
    assert_equal(as_int(2.1, False), 2)
    assert_equal(as_int(-2.1, False), -2)
    v = np.longdouble(2**64)
    assert_equal(as_int(v), 2**64)
    # Have all long doubles got 63+1 binary bits of precision?  Windows 32-bit
    # longdouble appears to have 52 bit precision, but we avoid that by checking
    # for known precisions that are less than that required
    try:
        nmant = type_info(np.longdouble)['nmant']
    except FloatingError:
        nmant = 63  # Unknown precision, let's hope it's at least 63
    v = np.longdouble(2) ** (nmant + 1) - 1
    assert_equal(as_int(v), 2**(nmant + 1) - 1)
    # Check for predictable overflow
    nexp64 = floor_log2(type_info(np.float64)['max'])
    with np.errstate(over='ignore'):
        val = np.longdouble(2**nexp64) * 2  # outside float64 range
    assert_raises(OverflowError, as_int, val)
    assert_raises(OverflowError, as_int, -val)


def test_int_to_float():
    # Convert python integer to floating point
    # Standard float types just return cast value
    for ie3 in IEEE_floats:
        nmant = type_info(ie3)['nmant']
        for p in range(nmant + 3):
            i = 2**p + 1
            assert_equal(int_to_float(i, ie3), ie3(i))
            assert_equal(int_to_float(-i, ie3), ie3(-i))
        # IEEEs in this case are binary formats only
        nexp = floor_log2(type_info(ie3)['max'])
        # Values too large for the format
        smn, smx = -2**(nexp + 1), 2**(nexp + 1)
        if ie3 is np.float64:
            assert_raises(OverflowError, int_to_float, smn, ie3)
            assert_raises(OverflowError, int_to_float, smx, ie3)
        else:
            assert_equal(int_to_float(smn, ie3), ie3(smn))
            assert_equal(int_to_float(smx, ie3), ie3(smx))
    # Longdoubles do better than int, we hope
    LD = np.longdouble
    # up to integer precision of float64 nmant, we get the same result as for
    # casting directly
    nmant = type_info(np.float64)['nmant']
    for p in range(nmant + 2):  # implicit
        i = 2**p - 1
        assert_equal(int_to_float(i, LD), LD(i))
        assert_equal(int_to_float(-i, LD), LD(-i))
    # Above max of float64, we're hosed
    nexp64 = floor_log2(type_info(np.float64)['max'])
    smn64, smx64 = -2**(nexp64 + 1), 2**(nexp64 + 1)
    # The algorithm here implemented goes through float64, so supermax and
    # supermin will cause overflow errors
    assert_raises(OverflowError, int_to_float, smn64, LD)
    assert_raises(OverflowError, int_to_float, smx64, LD)
    try:
        nmant = type_info(np.longdouble)['nmant']
    except FloatingError:  # don't know where to test
        return
    # test we recover precision just above nmant
    i = 2**(nmant + 1) - 1
    assert_equal(as_int(int_to_float(i, LD)), i)
    assert_equal(as_int(int_to_float(-i, LD)), -i)
    # Test no error for longs
    if PY2:
        i = long(i)
        assert_equal(as_int(int_to_float(i, LD)), i)
        assert_equal(as_int(int_to_float(-i, LD)), -i)
    # If longdouble can cope with 2**64, test
    if nmant >= 63:
        # Check conversion to int; the line below causes an error subtracting
        # ints / uint64 values, at least for Python 3.3 and numpy dev 1.8
        big_int = np.uint64(2**64 - 1)
        assert_equal(as_int(int_to_float(big_int, LD)), big_int)


def test_as_int_np_fix():
    # Test as_int works for integers.  We need as_int for integers because of a
    # numpy 1.4.1 bug such that int(np.uint32(2**32-1) == -1
    for t in np.sctypes['int'] + np.sctypes['uint']:
        info = np.iinfo(t)
        mn, mx = np.array([info.min, info.max], dtype=t)
        assert_equal((mn, mx), (as_int(mn), as_int(mx)))


def test_floor_exact_16():
    # A normal integer can generate an inf in float16
    if not have_float16:
        raise SkipTest('No float16')
    assert_equal(floor_exact(2**31, np.float16), np.inf)
    assert_equal(floor_exact(-2**31, np.float16), -np.inf)


def test_floor_exact_64():
    # float64
    for e in range(53, 63):
        start = np.float64(2**e)
        across = start + np.arange(2048, dtype=np.float64)
        gaps = set(np.diff(across)).difference([0])
        assert_equal(len(gaps), 1)
        gap = gaps.pop()
        assert_equal(gap, int(gap))
        test_val = 2**(e + 1) - 1
        assert_equal(floor_exact(test_val, np.float64), 2**(e + 1) - int(gap))


def test_floor_exact():
    to_test = IEEE_floats + [float]
    try:
        type_info(np.longdouble)['nmant']
    except FloatingError:
        # Significand bit count not reliable, don't test long double
        pass
    else:
        to_test.append(np.longdouble)
    # When numbers go above int64 - I believe, numpy comparisons break down,
    # so we have to cast to int before comparison
    int_flex = lambda x, t: as_int(floor_exact(x, t))
    int_ceex = lambda x, t: as_int(ceil_exact(x, t))
    for t in to_test:
        # A number bigger than the range returns the max
        info = type_info(t)
        assert_equal(floor_exact(2**5000, t), np.inf)
        assert_equal(ceil_exact(2**5000, t), np.inf)
        # A number more negative returns -inf
        assert_equal(floor_exact(-2**5000, t), -np.inf)
        assert_equal(ceil_exact(-2**5000, t), -np.inf)
        # Check around end of integer precision
        nmant = info['nmant']
        for i in range(nmant + 1):
            iv = 2**i
            # up to 2**nmant should be exactly representable
            for func in (int_flex, int_ceex):
                assert_equal(func(iv, t), iv)
                assert_equal(func(-iv, t), -iv)
                assert_equal(func(iv - 1, t), iv - 1)
                assert_equal(func(-iv + 1, t), -iv + 1)
        if t is np.longdouble and (
                on_powerpc() or
                longdouble_precision_improved()):
            # The nmant value for longdouble on PPC appears to be conservative,
            # so that the tests for behavior above the nmant range fail.
            # windows longdouble can change from float64 to Intel80 in some
            # situations, in which case nmant will not be correct
            continue
        # Confirm to ourselves that 2**(nmant+1) can't be exactly represented
        iv = 2**(nmant + 1)
        assert_equal(int_flex(iv + 1, t), iv)
        assert_equal(int_ceex(iv + 1, t), iv + 2)
        # negatives
        assert_equal(int_flex(-iv - 1, t), -iv - 2)
        assert_equal(int_ceex(-iv - 1, t), -iv)
        # The gap in representable numbers is 2 above 2**(nmant+1), 4 above
        # 2**(nmant+2), and so on.
        for i in range(5):
            iv = 2**(nmant + 1 + i)
            gap = 2**(i + 1)
            assert_equal(as_int(t(iv) + t(gap)), iv + gap)
            for j in range(1, gap):
                assert_equal(int_flex(iv + j, t), iv)
                assert_equal(int_flex(iv + gap + j, t), iv + gap)
                assert_equal(int_ceex(iv + j, t), iv + gap)
                assert_equal(int_ceex(iv + gap + j, t), iv + 2 * gap)
            # negatives
            for j in range(1, gap):
                assert_equal(int_flex(-iv - j, t), -iv - gap)
                assert_equal(int_flex(-iv - gap - j, t), -iv - 2 * gap)
                assert_equal(int_ceex(-iv - j, t), -iv)
                assert_equal(int_ceex(-iv - gap - j, t), -iv - gap)


def test_usable_binary128():
    # Check for usable binary128
    yes = have_binary128()
    with np.errstate(over='ignore'):
        exp_test = np.longdouble(2) ** 16383
    assert_equal(yes,
                 exp_test.dtype.itemsize == 16 and
                 np.isfinite(exp_test) and
                 _check_nmant(np.longdouble, 112))
