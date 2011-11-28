""" Investigate casting rules

Specifically, does the data affect the output type?

To do this, we combine all types and investigate the output, when:

* type A, B have data (0)
* type A has max / min for type A, type B has data (0)
* type A has max / min for type A, type B has max / min for type B

Expecting that in all cases the same dtype will result.

In fact what happens is that this _is_ true if A and B are atleast_1d, but it is
not true if (A or B is a scalar, for numpy 1.6.1). It looks like numpy 1.6.1 is
first checking whether the scalar B np.can_cast to type A, if so, then the
return type is type of A, otherwise it uses the array casting rules.

Thus - for numpy 1.6.1::

    >>> import numpy as np
    >>> Adata = np.array([127], dtype=np.int8)
    >>> Bdata = np.int16(127)
    >>> (Adata + Bdata).dtype
    dtype('int8')
    >>> Bdata = np.int16(128)
    >>> (Adata + Bdata).dtype
    dtype('int16')
    >>> Bdata = np.array([127], dtype=np.int16)
    >>> (Adata + Bdata).dtype
    dtype('int16')
"""

from distutils.version import LooseVersion

import numpy as np

from nose.tools import assert_equal

NP_VERSION = LooseVersion(np.__version__)

ALL_TYPES = (np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float'] +
             np.sctypes['complex'])

def get_info(type):
    try:
        return np.finfo(type)
    except ValueError:
        return np.iinfo(type)


def test_cast_assumptions():
    # Check that dtype is predictable from binary operations
    npa = np.array
    for A in ALL_TYPES:
        a_info = get_info(A)
        for B in ALL_TYPES:
            b_info = get_info(B)
            Adata = np.zeros((2,), dtype=A)
            Bdata = np.zeros((2,), dtype=B)
            Bscalar = B(0) # 0 can always be cast to type A
            out_dtype = (Adata + Bdata).dtype
            out_sc_dtype = (Adata + Bscalar).dtype
            assert_equal(out_dtype, (Adata * Bdata).dtype)
            assert_equal(out_sc_dtype, (Adata * Bscalar).dtype)
            Adata[0], Adata[1] = a_info.min, a_info.max
            assert_equal(out_dtype, (Adata + Bdata).dtype)
            assert_equal(out_dtype, (Adata * Bdata).dtype)
            # Compiled array gives same dtype
            assert_equal(out_dtype, npa([Adata[0:1], Bdata[0:1]]).dtype)
            assert_equal(out_sc_dtype, (Adata + Bscalar).dtype)
            assert_equal(out_sc_dtype, (Adata * Bscalar).dtype)
            # Compiled array with scalars gives promoted (can_cast) dtype
            assert_equal(out_dtype, npa([Adata[0], Bscalar]).dtype)
            Bdata[0], Bdata[1] = b_info.min, b_info.max
            Bscalar = B(b_info.max) # cannot always be cast to type A
            assert_equal(out_dtype, (Adata + Bdata).dtype)
            assert_equal(out_dtype, (Adata * Bdata).dtype)
            # Compiled array with scalars - promoted dtype
            assert_equal(out_dtype, npa([Adata[0], Bscalar]).dtype)
            # Here numpy >= 1.6.1 differs from previous versions
            if NP_VERSION <= '1.5.1' or np.can_cast(Bscalar, A):
                assert_equal(out_sc_dtype, (Adata + Bscalar).dtype)
                assert_equal(out_sc_dtype, (Adata * Bscalar).dtype)
            else: # casting rules changed for 1.6 onwards
                assert_equal(out_dtype, (Adata + Bscalar).dtype)
                assert_equal(out_dtype, (Adata * Bscalar).dtype)
