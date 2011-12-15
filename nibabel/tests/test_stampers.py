""" Testing state stamping
"""

import numpy as np

from ..py3k import ZEROB, asbytes

from ..stampers import Unknown, is_unknown, Stamper, NdaStamper

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_not_equal, assert_raises)


def test_uknown():
    # Unknown singleton-like
    u = Unknown()
    assert_equal(repr(u), 'Unknown()')
    assert_equal(str(u), 'Unknown()')
    assert_false(u == u)
    assert_true(u != u)
    assert_not_equal(u, u)
    p = Unknown()
    assert_not_equal(u, p)
    assert_true(is_unknown(u))
    assert_false(is_unknown(1))
    # Note - this _is_ equal
    # assert_not_equal((1, u), (1, u))


def test_stamper():
    # state_stamp can can from
    # * state_stamper() method
    # Some immutables -> themselves
    # Otherwise the signature is Unknown()
    class D(object): # Class for testing get_signature
        def state_stamper(self, cstate):
            return self.__class__, 28
    for ster in (Stamper(), NdaStamper()):
        assert_equal(ster(None), ster(None))
        assert_not_equal(ster(None), ster(1))
        assert_equal(ster(1), ster(1))
        assert_not_equal(ster(1), ster(2))
        assert_equal(ster('a string'), ster('a string'))
        assert_not_equal(ster('a string'), ster(1))
        bs = asbytes('some bytes')
        assert_equal(ster(bs), ster(bs))
        assert_equal(ster(1.0), ster(1.0))
        assert_not_equal(ster(1.0), ster(1))
        # an anonymous object, usually not stampable
        ob = object()
        assert_not_equal(ster(ob), ster(ob))
        L = [1, 2]
        T = (1, 2)
        assert_equal(ster(L), ster(L[:]))
        assert_equal(ster(T), ster(T[:]))
        assert_not_equal(ster(L), ster(T))
        assert_not_equal(ster((1, ob)), ster((1, ob)))
        d1 = D()
        d2 = D()
        assert_equal(ster(d1), ster(d2))
        # Dictionaries
        di1 = dict(a = 1, b = 2)
        di2 = dict(a = 1, b = 2)
        assert_equal(ster(di1), ster(di2))
        # They are not just defined by their items, but by their type
        assert_not_equal(ster(di1), ster(di2.items()))
        # Inherited types don't work because they might have more state
        class MyList(list): pass
        class MyTuple(tuple): pass
        class MyDict(dict): pass
        assert_not_equal(ster(MyList((1,2))), ster(MyList((1,2))))
        assert_not_equal(ster(MyTuple((1,2))), ster(MyTuple((1,2))))
        assert_not_equal(ster(MyDict(a=1, b=2)), ster(MyDict(a=1, b=2)))
        # Classes pass through, even if they have state_stamper methods
        assert_equal(ster(D), ster(D))


def test_nda_stamper():
    # Arrays work if they are small
    nda_ster = NdaStamper()
    arr1 = np.zeros((3,), dtype=np.int16)
    arr2 = np.zeros((3,), dtype=np.int16)
    assert_equal(nda_ster(arr1), nda_ster(arr2))
    # The data has to be the same
    arr2p1 = arr2.copy()
    arr2p1[0] = 1
    assert_not_equal(nda_ster(arr1), nda_ster(arr2p1))
    # Comparison depends on the byte threshold
    nda_ster5 = NdaStamper(byte_thresh = 5)
    assert_not_equal(nda_ster5(arr1), nda_ster5(arr1))
    # Byte thresh gets passed down to iterations of lists
    assert_equal(nda_ster([1, arr1]), nda_ster([1, arr2]))
    assert_not_equal(nda_ster5([1, arr1]), nda_ster5([1, arr1]))
    # Arrays in dicts
    d1 = dict(a = 1, b = arr1)
    d2 = dict(a = 1, b = arr2)
    assert_equal(nda_ster(d1), nda_ster(d2))
    # Byte thresh gets passed down to iterations of dicts
    assert_not_equal(nda_ster5(d1), nda_ster5(d1))
    # Make sure strings distinguished from arrays
    bs = asbytes('byte string')
    sarr = np.array(bs, dtype = 'S')
    assert_equal(nda_ster(sarr), nda_ster(sarr.copy()))
    assert_not_equal(nda_ster(sarr), nda_ster(bs))
    # shape and dtype also distinguished
    arr3 = arr2.reshape((1,3))
    assert_not_equal(nda_ster(arr1), nda_ster(arr3))
    arr4 = arr3.reshape((3,))
    assert_equal(nda_ster(arr1), nda_ster(arr4))
    arr5 = arr1.newbyteorder('s')
    assert_array_equal(arr1, arr5)
    assert_not_equal(nda_ster(arr1), nda_ster(arr5))
    arr6 = arr5.newbyteorder('s')
    assert_equal(nda_ster(arr1), nda_ster(arr6))
