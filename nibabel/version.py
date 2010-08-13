""" This file contains defines parameters for nibabel that we use to fill
settings in setup.py, the nibabel top-level docstring, and for building the
docs.  In setup.py in particular, we exec this file, so it cannot import nibabel
"""

# nibabel version information.  An empty _version_extra corresponds to a
# full release.  '-dev' as a _version_extra string means this is a development
# version, and we append the git commit hash short form so we know what code
# state this is.
_version_major = 0
_version_minor = 9
_version_micro = 0
_version_extra = '-dev'

# The next line allows 'git archive' to dump the tag into this string.  This
# should never happen to the file while still in the git repository.
_maybe_subst_hash = '$Format:%h$'

def get_commit_hash():
    ''' Get short form of commit hash for appending to _version_extra

    We get this from (in order of preference):

    * A substituted value in ``_maybe_subst_hash``
    * The text in a file '_commit_hash.txt' in the same directory as this file
      (iff we are being imported)
    * git's output, if we are in a git repository

    Otherwise we return the empty string
    '''
    if not _maybe_subst_hash.startswith('$Format'): # it has been substituted
        return _maybe_subst_hash
    if __name__ == '__main__':
        # we are being executed probably from setup.py
        cwd = None
    else:
        # we are being imported
        import os
        cwd = os.path.dirname(__file__)
        # Try and get commit from written commit text file
        pth = os.path.join(cwd, '_commit_hash.txt')
        if os.path.isfile(pth):
            return file(pth).read().strip()
    # maybe we are in a repository
    import subprocess
    proc = subprocess.Popen('git rev-parse --short HEAD',
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            cwd=cwd, shell=True)
    repo_commit, _ = proc.communicate()
    if repo_commit:
        return repo_commit.strip()
    return ''


# If the version extra above is '-dev' we need to append the short form of the
# commit hash.
if _version_extra == '-dev':
    _version_extra += get_commit_hash()

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
