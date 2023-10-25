import numpy as np
import pytest

# Ignore warning requesting help with nicom
with pytest.warns(UserWarning):
    import nibabel.nicom


def pytest_configure(config):
    """Configure pytest options."""
    if int(np.__version__[0]) >= 2:
        np.set_printoptions(legacy=125)
