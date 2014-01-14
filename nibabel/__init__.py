# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import os

from .info import __version__, long_description as __doc__
__doc__ += """
Quickstart
==========

::

   import nibabel as nib

   img1 = nib.load('my_file.nii')
   img2 = nib.load('other_file.nii.gz')
   img3 = nib.load('spm_file.img')

   data = img1.get_data()
   affine = img1.get_affine()

   print img1

   nib.save(img1, 'my_file_copy.nii.gz')

   new_image = nib.Nifti1Image(data, affine)
   nib.save(new_image, 'new_image.nii.gz')

For more detailed information see the :ref:`manual`.
"""
# module imports
from . import analyze as ana
from . import spm99analyze as spm99
from . import spm2analyze as spm2
from . import nifti1 as ni1
from . import ecat
# object imports
from .fileholders import FileHolder, FileHolderError
from .loadsave import load, save
from .arrayproxy import is_proxy
from .analyze import AnalyzeHeader, AnalyzeImage
from .spm99analyze import Spm99AnalyzeHeader, Spm99AnalyzeImage
from .spm2analyze import Spm2AnalyzeHeader, Spm2AnalyzeImage
from .nifti1 import Nifti1Header, Nifti1Image, Nifti1Pair
from .nifti2 import Nifti2Header, Nifti2Image, Nifti2Pair
from .minc1 import Minc1Image
from .minc2 import Minc2Image
# Deprecated backwards compatiblity for MINC1
from .deprecated import ModuleProxy as _ModuleProxy
minc = _ModuleProxy('nibabel.minc')
from .minc1 import MincImage
from .freesurfer import MGHImage
from .funcs import (squeeze_image, concat_images, four_to_three,
                    as_closest_canonical)
from .orientations import (io_orientation, orientation_affine,
                           flip_axis, OrientationError,
                           apply_orientation, aff2axcodes)
from .imageclasses import class_map, ext_map
from . import trackvis

# be friendly on systems with ancient numpy -- no tests, but at least
# importable
try:
    from numpy.testing import Tester
    test = Tester().test
    bench = Tester().bench
    del Tester
except ImportError:
    def test(*args, **kwargs): raise RuntimeError('Need numpy >= 1.2 for tests')

from .pkg_info import get_pkg_info as _get_pkg_info
get_info = lambda : _get_pkg_info(os.path.dirname(__file__))
