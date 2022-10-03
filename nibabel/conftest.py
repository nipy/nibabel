import pytest

# Pre-load deprecated modules to avoid cluttering warnings
with pytest.warns(FutureWarning):
    import nibabel.py3k

# Ignore warning requesting help with nicom
with pytest.warns(UserWarning):
    import nibabel.nicom
