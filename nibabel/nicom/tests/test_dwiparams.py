""" Testing diffusion parameter processing

"""

import numpy as np

from ..dwiparams import B2q

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_b2q():
    # conversion of b matrix to q
    q = np.array([1,2,3])
    s = np.sqrt(np.sum(q * q)) # vector norm
    B = np.outer(q, q)
    assert_array_almost_equal(q*s, B2q(B))
    q = np.array([1,2,3])
    # check that the sign of the vector as positive x convention
    B = np.outer(-q, -q)
    assert_array_almost_equal(q*s, B2q(B))
    q = np.array([-1, 2, 3])
    B = np.outer(q, q)
    assert_array_almost_equal(-q*s, B2q(B))
    # Massive negative eigs
    B = np.eye(3) * -1
    assert_raises(ValueError, B2q, B)
    # no error if we up the tolerance
    q = B2q(B, tol=1)
    # Less massive negativity, dropping tol
    B = np.diag([-1e-14, 10., 1])
    assert_raises(ValueError, B2q, B)
    assert_array_almost_equal(B2q(B, tol=5e-13), [0, 10, 0])
    # Confirm that we assume symmetric
    B = np.eye(3)
    B[0, 1] = 1e-5
    assert_raises(ValueError, B2q, B)
