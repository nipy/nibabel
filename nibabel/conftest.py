import pytest


@pytest.fixture(autouse=True, scope="session")
def set_printopts():
    import numpy as np
    from distutils.version import LooseVersion

    if LooseVersion(np.__version__) >= LooseVersion("1.14"):
        legacy_printopt = np.get_printoptions().get("legacy")
        np.set_printoptions(legacy="1.13")
        yield
        np.set_printoptions(legacy=legacy_printopt)
    else:
        yield
