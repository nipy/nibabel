import numpy as np
import pytest

from packaging.version import parse

# Ignore warning requesting help with nicom
with pytest.warns(UserWarning):
    import nibabel.nicom


def pytest_configure(config):
    """Configure pytest options."""
    if parse('1.26') <= parse(np.__version__):
        np.set_printoptions(legacy='1.25')
