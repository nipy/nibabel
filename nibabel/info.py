""" This file contains defines parameters for nibabel that we use to fill
settings in setup.py, the nibabel top-level docstring, and for building the
docs.  In setup.py in particular, we exec this file, so it cannot import nibabel
"""

# nibabel version information.  An empty _version_extra corresponds to a
# full release.  '.dev' as a _version_extra string means this is a development
# version
_version_major = 2
_version_minor = 1
_version_micro = 0
_version_extra = 'dev'
#_version_extra = ''

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
__version__ = "%s.%s.%s%s" % (_version_major,
                              _version_minor,
                              _version_micro,
                              _version_extra)

CLASSIFIERS = ["Development Status :: 4 - Beta",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description  = 'Access a multitude of neuroimaging data formats'

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
PAR/REC.  We can read and write Freesurfer_ geometry, and read Freesurfer
morphometry and annotation files.  There is some very limited support for
DICOM_.  NiBabel is the successor of PyNIfTI_.

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

The various image format classes give full or selective access to header (meta)
information and access to the image data is made available via NumPy arrays.

Website
=======

Current documentation on nibabel can always be found at the `NIPY nibabel
website <https://nipy.github.io/nibabel>`_.

Mailing Lists
=============

Please see the `nipy devel list
<http://mail.scipy.org/mailman/listinfo/nipy-devel>`_. The nipy devel list is
fine for user and developer questions about nibabel.

Code
====

You can find our sources and single-click downloads:

* `Main repository`_ on Github;
* Documentation_ for all releases and current development tree;
* Download the `current release`_ from pypi;
* Download `current development version`_ as a zip file;
* Downloads of all `available releases`_.

.. _main repository: https://github.com/nipy/nibabel
.. _Documentation: https://nipy.github.io/nibabel
.. _current release: https://pypi.python.org/pypi/nibabel
.. _current development version: https://github.com/nipy/nibabel/archive/master.zip
.. _available releases: https://github.com/nipy/nibabel/releases

License
=======

Nibabel is licensed under the terms of the MIT license. Some code included with
nibabel is licensed under the BSD license.  Please see the COPYING file in the
nibabel distribution.
"""

# versions for dependencies
NUMPY_MIN_VERSION='1.5'
PYDICOM_MIN_VERSION='0.9.7'

# Main setup parameters
NAME                = 'nibabel'
MAINTAINER          = "Matthew Brett, Michael Hanke and Eric Larson"
MAINTAINER_EMAIL    = "nipy-devel@neuroimaging.scipy.org"
DESCRIPTION         = description
LONG_DESCRIPTION    = long_description
URL                 = "https://nipy.github.io/nibabel"
DOWNLOAD_URL        = "https://github.com/nipy/nibabel"
LICENSE             = "MIT license"
CLASSIFIERS         = CLASSIFIERS
AUTHOR              = "Matthew Brett, Michael Hanke, Stephan Gerhard"
AUTHOR_EMAIL        = "nipy-devel@neuroimaging.scipy.org"
PLATFORMS           = "OS Independent"
MAJOR               = _version_major
MINOR               = _version_minor
MICRO               = _version_micro
ISRELEASE           = _version_extra == ''
VERSION             = __version__
PROVIDES            = ["nibabel", 'nisext']
REQUIRES            = ["numpy (>=%s)" % NUMPY_MIN_VERSION]
