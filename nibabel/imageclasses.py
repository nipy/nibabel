# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Define supported image classes and names '''
import warnings

from .analyze import AnalyzeImage
from .freesurfer import MGHImage
from .gifti import GiftiImage
from .minc1 import Minc1Image
from .minc2 import Minc2Image
from .nifti1 import Nifti1Pair, Nifti1Image
from .nifti2 import Nifti2Pair, Nifti2Image
from .parrec import PARRECImage
from .spm99analyze import Spm99AnalyzeImage
from .spm2analyze import Spm2AnalyzeImage
from .volumeutils import Recoder

from .optpkg import optional_package
_, have_scipy, _ = optional_package('scipy')


# Ordered by the load/save priority.
all_image_classes = [Nifti1Pair, Nifti1Image, Nifti2Pair, Nifti2Image,
                     Spm2AnalyzeImage, Spm99AnalyzeImage, AnalyzeImage,
                     Minc1Image, Minc2Image, MGHImage,
                     PARRECImage, GiftiImage]


# DEPRECATED: mapping of names to classes and class functionality
class ClassMapDict(dict):
    def __getitem__(self, *args, **kwargs):
        warnings.warn("class_map is deprecated.", DeprecationWarning,
                      stacklevel=2)
        return super(ClassMapDict, self).__getitem__(*args, **kwargs)

class_map = ClassMapDict(
    analyze={'class': AnalyzeImage,  # Image class
             'ext': '.img',  # characteristic image extension
             'has_affine': False,  # class can store an affine
             'makeable': True,  # empty image can be easily made in memory
             'rw': True},  # image can be written
    spm99analyze={'class': Spm99AnalyzeImage,
                  'ext': '.img',
                  'has_affine': True,
                  'makeable': True,
                  'rw': have_scipy},
    spm2analyze={'class': Spm2AnalyzeImage,
                 'ext': '.img',
                 'has_affine': True,
                 'makeable': True,
                 'rw': have_scipy},
    nifti_pair={'class': Nifti1Pair,
                'ext': '.img',
                'has_affine': True,
                'makeable': True,
                'rw': True},
    nifti_single={'class': Nifti1Image,
                  'ext': '.nii',
                  'has_affine': True,
                  'makeable': True,
                  'rw': True},
    minc={'class': Minc1Image,
          'ext': '.mnc',
          'has_affine': True,
          'makeable': True,
          'rw': False},
    mgh={'class': MGHImage,
         'ext': '.mgh',
         'has_affine': True,
         'makeable': True,
         'rw': True},
    mgz={'class': MGHImage,
         'ext': '.mgz',
         'has_affine': True,
         'makeable': True,
         'rw': True},
    par={'class': PARRECImage,
         'ext': '.par',
         'has_affine': True,
         'makeable': False,
         'rw': False})


class ExtMapRecoder(Recoder):
    def __getitem__(self, *args, **kwargs):
        warnings.warn("ext_map is deprecated.", DeprecationWarning,
                      stacklevel=2)
        return super(ExtMapRecoder, self).__getitem__(*args, **kwargs)

# mapping of extensions to default image class names
ext_map = ExtMapRecoder((
    ('nifti_single', '.nii'),
    ('nifti_pair', '.img', '.hdr'),
    ('minc', '.mnc'),
    ('mgh', '.mgh'),
    ('mgz', '.mgz'),
    ('par', '.par'),
))
