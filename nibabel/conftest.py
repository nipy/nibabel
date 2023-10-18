import numpy as np
import pytest

# Ignore warning requesting help with nicom
with pytest.warns(UserWarning):
    import nibabel.nicom


def pytest_configure(config):
    """Configure pytest options."""
    np.set_printoptions(legacy=125)
