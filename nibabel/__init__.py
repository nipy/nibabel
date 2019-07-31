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
   affine = img1.affine

   print(img1)

   nib.save(img1, 'my_file_copy.nii.gz')

   new_image = nib.Nifti1Image(data, affine)
   nib.save(new_image, 'new_image.nii.gz')

For more detailed information see the :ref:`manual`.
"""

# Package-wide test setup and teardown
_test_states = {
    # Numpy changed print options in 1.14; we can update docstrings and remove
    # these when our minimum for building docs exceeds that
    'legacy_printopt': None,
    }

def setup_package():
    """ Set numpy print style to legacy="1.13" for newer versions of numpy """
    import numpy as np
    from distutils.version import LooseVersion
    if LooseVersion(np.__version__) >= LooseVersion('1.14'):
        if _test_states.get('legacy_printopt') is None:
            _test_states['legacy_printopt'] = np.get_printoptions().get('legacy')
        np.set_printoptions(legacy="1.13")

def teardown_package():
    """ Reset print options when tests finish """
    import numpy as np
    if _test_states.get('legacy_printopt') is not None:
        np.set_printoptions(legacy=_test_states.pop('legacy_printopt'))


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
from .cifti2 import Cifti2Header, Cifti2Image
from .gifti import GiftiImage
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
from .imageclasses import class_map, ext_map, all_image_classes
trackvis = _ModuleProxy('nibabel.trackvis')
from . import mriutils
from . import streamlines
from . import viewers

import pkgutil

if not pkgutil.find_loader('mock'):
    def test(*args, **kwargs):
        raise RuntimeError('Need "mock" package for tests')
else:
    from numpy.testing import Tester
    test = Tester().test
    bench = Tester().bench
    del Tester

del pkgutil

from .pkg_info import get_pkg_info as _get_pkg_info


def get_info():
    return _get_pkg_info(os.path.dirname(__file__))
