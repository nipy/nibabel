# init for sext package
"""Setuptools extensions

nibabel uses these routines, and houses them, and installs them.  nipy-proper
and dipy use them.
"""

import warnings

warnings.warn(
    """The nisext package is deprecated as of NiBabel 5.0 and will be fully
removed in NiBabel 6.0"""
)
