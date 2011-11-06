""" Test floating point deconstructions and floor methods
"""
import numpy as np

from ..casting import floor_exact, flt2nmant, as_int, FloatingError

from nose import SkipTest
from nose.tools import assert_equal, assert_raises

IEEE_floats = [np.float32, np.float64]
try:
    np.float16
except AttributeError: # float16 not present in np < 1.6
    have_float16 = False
else:
    have_float16 = True
if have_float16:
    IEEE_floats.append(np.float16)

LD_INFO = np.finfo(np.longdouble)

def test_flt2nmant():
    for t in IEEE_floats:
        assert_equal(flt2nmant(t), np.finfo(t).nmant)
    if (LD_INFO.nmant, LD_INFO.nexp) == (63, 15):
        assert_equal(flt2nmant(np.longdouble), 63)


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
    # Have all long doubles got this precision?  We'll see I guess
    assert_equal(as_int(v+2), 2**64+2)


def test_floor_exact_16():
    # A normal integer can generate an inf in float16
    if not have_float16:
        raise SkipTest('No float16')
    assert_equal(floor_exact(2**31, np.float16), np.finfo(np.float16).max)


def test_floor_exact_64():
    # float64
    for e in range(53, 63):
        start = np.float64(2**e)
        across = start + np.arange(2048, dtype=np.float64)
        gaps = set(np.diff(across)).difference([0])
        assert_equal(len(gaps), 1)
        gap = gaps.pop()
        assert_equal(gap, int(gap))
        test_val = 2**(e+1)-1
        assert_equal(floor_exact(test_val, np.float64), 2**(e+1)-int(gap))


def test_floor_exact():
    to_test = IEEE_floats + [float]
    try:
        flt2nmant(np.longdouble)
    except FloatingError:
        # Significand bit count not reliable, don't test long double
        pass
    else:
        to_test.append(np.longdouble)
    # When numbers go above int64 - I believe, numpy comparisons break down,
    # so we have to cast to int before comparison
    int_flex = lambda x, t : as_int(floor_exact(x, t))
    for t in to_test:
        # A number bigger than the range returns the max
        assert_equal(floor_exact(2**5000, t), np.finfo(t).max)
        nmant = flt2nmant(t)
        for i in range(nmant+1):
            iv = 2**i
            # up to 2**nmant should be exactly representable
            assert_equal(int_flex(iv, t), iv)
            assert_equal(int_flex(-iv, t), -iv)
            assert_equal(int_flex(iv-1, t), iv-1)
            assert_equal(int_flex(-iv+1, t), -iv+1)
        # 2**(nmant+1) can't be exactly represented
        iv = 2**(nmant+1)
        assert_equal(int_flex(iv+1, t), iv)
        # negatives
        assert_equal(int_flex(-iv-1, t), -iv)
        # The gap in representable numbers is 2 above 2**(nmant+1), 4 above
        # 2**(nmant+2), and so on.
        for i in range(5):
            iv = 2**(nmant+1+i)
            gap = 2**(i+1)
            assert_equal(as_int(t(iv) + t(gap)), iv+gap)
            for j in range(1,gap):
                assert_equal(int_flex(iv+j, t), iv)
                assert_equal(int_flex(iv+gap+j, t), iv+gap)
            # negatives
            for j in range(1,gap):
                assert_equal(int_flex(-iv-j, t), -iv)
                assert_equal(int_flex(-iv-gap-j, t), -iv-gap)
