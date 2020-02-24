"""
These tests are almost certainly overkill, but serve to verify that
the behavior of _h5py_compat is pass-through in all but a small set of
well-defined cases
"""
import sys
import os
from distutils.version import LooseVersion
import numpy as np

from ..optpkg import optional_package
from .. import _h5py_compat as compat

h5py, have_h5py, _ = optional_package('h5py')


def test_optpkg_equivalence():
    # No effect on Linux/OSX
    if os.name == 'posix':
        assert have_h5py == compat.have_h5py
    # No effect on Python 2.7 or 3.6+
    if sys.version_info >= (3, 6) or sys.version_info < (3,):
        assert have_h5py == compat.have_h5py
    # Available in a strict subset of cases
    if not have_h5py:
        assert not compat.have_h5py
    # Available when version is high enough
    elif LooseVersion(h5py.__version__) >= '2.10':
        assert compat.have_h5py


def test_disabled_h5py_cases():
    # On mismatch
    if have_h5py and not compat.have_h5py:
        # Recapitulate min_h5py conditions from _h5py_compat
        assert os.name == 'nt'
        assert (3,) <= sys.version_info < (3, 6)
        assert LooseVersion(h5py.__version__) < '2.10'
        # Verify that the root cause is present
        # If any tests fail, they will likely be these, so they may be
        # ill-advised...
        if LooseVersion(np.__version__) < '1.18':
            assert str(np.longdouble) == str(np.float64)
        else:
            assert str(np.longdouble) != str(np.float64)
        assert np.longdouble != np.float64
