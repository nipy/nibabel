#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""This module provides Python bindings to the NIfTI data format.

The PyNIfTI module is a Python interface to the NIfTI I/O libraries. Using
PyNIfTI, one can easily read and write NIfTI and ANALYZE images from within
Python. The :class:`~nifti.image.NiftiImage` class provides pythonic
access to the full header information and for a maximum of interoperability the
image data is made available via NumPy arrays.
"""

__docformat__ = 'restructuredtext'

# expose the two main classes
from nifti.image import NiftiImage, MemMappedNiftiImage

# canonical version string
__version__ = '0.20100412.1'
