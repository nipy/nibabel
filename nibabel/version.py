""" This file contains defines parameters for nibabel that we use to fill
settings in setup.py, the nibabel top-level docstring, and for building the
docs.  In setup.py in particular, we exec this file, so it cannot import nibabel
"""

# nibabel version information.  An empty _version_extra corresponds to a release
_version_major = 0
_version_minor = 9
_version_micro = 0
_version_extra = '-dev'

# If the version extra above is '-dev' we need to append the short form of the
# commit hash.
if _version_extra == '-dev':
    # The next line allows git archive to dump the tag into this
    # string.  This should never happen to the file while still in the git
    # repository.
    archived_commit = '$Format:%h$'
    if not archived_commit.startswith('$Format'): # it has been substituted
        _version_extra += archived_commit
    elif __name__ != '__main__': # we're being imported rather than exec'ed
        # we might have been installed from a repository directory, in which
        # case we look for a generated commit hash text file
        import os
        cwd = os.path.dirname(__file__)
        pth = os.path.join(cwd, '_commit_hash.txt')
        if os.path.isfile(pth):
            repo_commit = file(pth).read().strip()
            _version_extra += repo_commit
        else: # maybe we are in a repository
            import subprocess
            proc = subprocess.Popen('git rev-parse --short HEAD',
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    cwd=cwd, shell=True)
            repo_commit, _ = proc.communicate()
            if repo_commit:
                repo_commit = repo_commit.strip()
                _version_extra += repo_commit

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

description  = 'Access a multitude of neuroimaging data formats',

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
ISRELEASED          = False
VERSION             = __version__
REQUIRES            = ["numpy"]
