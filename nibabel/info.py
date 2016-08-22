""" Define distrubution parameters for nibabel, including package version

This file contains defines parameters for nibabel that we use to fill settings
in setup.py, the nibabel top-level docstring, and for building the docs.  In
setup.py in particular, we exec this file, so it cannot import nibabel.
"""

import re
from distutils.version import StrictVersion

# nibabel version information.  An empty _version_extra corresponds to a
# full release.  *Any string* in `_version_extra` labels the version as
# pre-release.  So, if `_version_extra` is not empty, the version is taken to
# be earlier than the same version where `_version_extra` is empty (see
# `cmp_pkg_version` below).
#
# We usually use `dev` as `_version_extra` to label this as a development
# (pre-release) version.
_version_major = 2
_version_minor = 1
_version_micro = 0
# _version_extra = 'dev'
_version_extra = ''

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
__version__ = "%s.%s.%s%s" % (_version_major,
                              _version_minor,
                              _version_micro,
                              _version_extra)


def _parse_version(version_str):
    """ Parse version string `version_str` in our format
    """
    match = re.match('([0-9.]*\d)(.*)', version_str)
    if match is None:
        raise ValueError('Invalid version ' + version_str)
    return match.groups()


def _cmp(a, b):
    """ Implementation of ``cmp`` for Python 3
    """
    return (a > b) - (a < b)


def cmp_pkg_version(version_str, pkg_version_str=__version__):
    """ Compare `version_str` to current package version

    To be valid, a version must have a numerical major version followed by a
    dot, followed by a numerical minor version.  It may optionally be followed
    by a dot and a numerical micro version, and / or by an "extra" string.
    *Any* extra string labels the version as pre-release, so `1.2.0somestring`
    compares as prior to (pre-release for) `1.2.0`, where `somestring` can be
    any string.

    Parameters
    ----------
    version_str : str
        Version string to compare to current package version
    pkg_version_str : str, optional
        Version of our package.  Optional, set fom ``__version__`` by default.

    Returns
    -------
    version_cmp : int
        1 if `version_str` is a later version than `pkg_version_str`, 0 if
        same, -1 if earlier.

    Examples
    --------
    >>> cmp_pkg_version('1.2.1', '1.2.0')
    1
    >>> cmp_pkg_version('1.2.0dev', '1.2.0')
    -1
    """
    version, extra = _parse_version(version_str)
    pkg_version, pkg_extra = _parse_version(pkg_version_str)
    if version != pkg_version:
        return _cmp(StrictVersion(version), StrictVersion(pkg_version))
    return (0 if extra == pkg_extra
            else 1 if extra == ''
            else -1 if pkg_extra == ''
            else _cmp(extra, pkg_extra))


CLASSIFIERS = ["Development Status :: 4 - Beta",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description = 'Access a multitude of neuroimaging data formats'

# Note: this long_description is the canonical place to edit this text.
# It also appears in README.rst, but it should get there by running
# ``tools/refresh_readme.py`` which pulls in this version.
# We also include this text in the docs by ``..include::`` in
# ``docs/source/index.rst``.
long_description = """
=======
NiBabel
=======

Read / write access to some common neuroimaging file formats

This package provides read +/- write access to some common medical and
neuroimaging file formats, including: ANALYZE_ (plain, SPM99, SPM2 and later),
GIFTI_, NIfTI1_, NIfTI2_, MINC1_, MINC2_, MGH_ and ECAT_ as well as Philips
PAR/REC.  We can read and write FreeSurfer_ geometry, annotation and
morphometry files.  There is some very limited support for DICOM_.  NiBabel is
the successor of PyNIfTI_.

.. _ANALYZE: http://www.grahamwideman.com/gw/brain/analyze/formatdoc.htm
.. _NIfTI1: http://nifti.nimh.nih.gov/nifti-1/
.. _NIfTI2: http://nifti.nimh.nih.gov/nifti-2/
.. _MINC1:
    https://en.wikibooks.org/wiki/MINC/Reference/MINC1_File_Format_Reference
.. _MINC2:
    https://en.wikibooks.org/wiki/MINC/Reference/MINC2.0_File_Format_Reference
.. _PyNIfTI: http://niftilib.sourceforge.net/pynifti/
.. _GIFTI: https://www.nitrc.org/projects/gifti
.. _MGH: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/MghFormat
.. _ECAT: http://xmedcon.sourceforge.net/Docs/Ecat
.. _Freesurfer: https://surfer.nmr.mgh.harvard.edu
.. _DICOM: http://medical.nema.org/

The various image format classes give full or selective access to header
(meta) information and access to the image data is made available via NumPy
arrays.

Website
=======

Current documentation on nibabel can always be found at the `NIPY nibabel
website <http://nipy.org/nibabel>`_.

Mailing Lists
=============

Please send any questions or suggestions to the `neuroimaging mailing list
<https://mail.python.org/mailman/listinfo/neuroimaging>`_.

Code
====

Install nibabel with::

    pip install nibabel

You may also be interested in:

* the `nibabel code repository`_ on Github;
* documentation_ for all releases and current development tree;
* download the `current release`_ from pypi;
* download `current development version`_ as a zip file;
* downloads of all `available releases`_.

.. _nibabel code repository: https://github.com/nipy/nibabel
.. _Documentation: http://nipy.org/nibabel
.. _current release: https://pypi.python.org/pypi/nibabel
.. _current development version: https://github.com/nipy/nibabel/archive/master.zip
.. _available releases: https://github.com/nipy/nibabel/releases

License
=======

Nibabel is licensed under the terms of the MIT license. Some code included
with nibabel is licensed under the BSD license.  Please see the COPYING file
in the nibabel distribution.
"""

# versions for dependencies. Check these against:
# doc/source/installation.rst
# requirements.txt
# .travis.yml
NUMPY_MIN_VERSION = '1.5.1'
PYDICOM_MIN_VERSION = '0.9.7'

# Main setup parameters
NAME = 'nibabel'
MAINTAINER = "Matthew Brett, Michael Hanke, Eric Larson, Chris Markiewicz"
MAINTAINER_EMAIL = "neuroimaging@python.org"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://nipy.org/nibabel"
DOWNLOAD_URL = "https://github.com/nipy/nibabel"
LICENSE = "MIT license"
CLASSIFIERS = CLASSIFIERS
AUTHOR = "nibabel developers"
AUTHOR_EMAIL = "neuroimaging@python.org"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
ISRELEASE = _version_extra == ''
VERSION = __version__
PROVIDES = ["nibabel", 'nisext']
REQUIRES = ["numpy (>=%s)" % NUMPY_MIN_VERSION]
