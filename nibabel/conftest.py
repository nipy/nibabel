import pytest

# Ignore warning requesting help with nicom
with pytest.warns(UserWarning):
    import nibabel.nicom
