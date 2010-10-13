""" This file contains defines parameters for nibabel that we use to fill
settings in setup.py, the nibabel top-level docstring, and for building the
docs.  In setup.py in particular, we exec this file, so it cannot import nibabel
"""

# nibabel version information.  An empty _version_extra corresponds to a
# full release.  '.dev' as a _version_extra string means this is a development
# version
_version_major = 1
_version_minor = 0
_version_micro = 0
_version_extra = '.dev'

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
__version__ = "%s.%s.%s%s" % (_version_major,
                              _version_minor,
                              _version_micro,
                              _version_extra)

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description  = 'Access a multitude of neuroimaging data formats'

# Note: this long_description is actually a copy/paste from the top-level
# README.txt, so that it shows up nicely on PyPI.  So please remember to edit
# it only in one place and sync it correctly.
long_description = """
=======
NiBabel
=======

This package provides read and write access to some common medical and
neuroimaging file formats, including: ANALYZE_ (plain, SPM99, SPM2),
GIFTI_, NIfTI1_, MINC_, as well as PAR/REC. NiBabel is the successor of
PyNIfTI_.

.. _ANALYZE: http://www.grahamwideman.com/gw/brain/analyze/formatdoc.htm
.. _NIfTI1: http://nifti.nimh.nih.gov/nifti-1/
.. _MINC: http://wiki.bic.mni.mcgill.ca/index.php/MINC
.. _PyNIfTI: http://niftilib.sourceforge.net/pynifti/
.. _GIFTI: http://www.nitrc.org/projects/gifti

The various image format classes give full or selective access to header (meta)
information and access to the image data is made available via NumPy arrays.
"""

# versions for dependencies
NUMPY_MIN_VERSION='1.2'
PYDICOM_MIN_VERSION='0.9.5'

# Main setup parameters
NAME                = 'nibabel'
MAINTAINER          = "Matthew Brett and Michael Hanke"
MAINTAINER_EMAIL    = "nipy-devel@neuroimaging.scipy.org"
DESCRIPTION         = description
LONG_DESCRIPTION    = long_description
URL                 = "http://nipy.org/nibabel"
DOWNLOAD_URL        = "http://github.com/nipy/nibabel/archives/master"
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
PROVIDES            = ["nibabel"]
REQUIRES            = ["numpy (>=%s)" % NUMPY_MIN_VERSION]
