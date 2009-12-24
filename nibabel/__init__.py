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

===============================
 pynifti python implementation
===============================

Quickstart::

   import nibabel

   img1 = nibabel.load('my_file.nii')
   img2 = nibabel.load('other_file.nii.gz')
   img3 = nibabel.load('spm_file.img')

   data = img1.get_data()
   affine = img1.get_affine()

   print img1

   nibabel.save(img1, 'my_file_copy.nii.gz')

   new_image = nibabel.Nifti1Image(data, affine)
   nibabel.save(new_image, 'new_image.nii.gz')
"""

__docformat__ = 'restructuredtext'

# canonical version string
__version__ = '1.0.0'


# module imports
from nibabel import analyze as ana
from nibabel import spm99analyze as spm99
from nibabel import spm2analyze as spm2
from nibabel import nifti1 as ni1
from nibabel import minc
# object imports
from nibabel.loadsave import load, save
from nibabel.analyze import AnalyzeHeader, AnalyzeImage
from nibabel.spm99analyze import Spm99AnalyzeHeader, Spm99AnalyzeImage
from nibabel.spm2analyze import Spm2AnalyzeHeader, Spm2AnalyzeImage
from nibabel.nifti1 import Nifti1Header, Nifti1Image
from nibabel.minc import MincHeader, MincImage
from nibabel.funcs import squeeze_image, concat_images, four_to_three
