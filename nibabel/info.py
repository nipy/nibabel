"""Define distribution parameters for nibabel, including package version

The long description parameter is used to fill settings in setup.py, the
nibabel top-level docstring, and in building the docs.
We exec this file in several places, so it cannot import nibabel or use
relative imports.
"""

# nibabel version information
# This is a fall-back for versioneer when installing from a git archive.
# This should be set to the intended next version + dev to indicate a
# development (pre-release) version.
_version_major = 5
_version_minor = 0
_version_micro = 0
_version_extra = '.dev0'

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
VERSION = f'{_version_major}.{_version_minor}.{_version_micro}{_version_extra}'
