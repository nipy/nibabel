# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Define supported image classes and names '''
from .analyze import AnalyzeImage
from .spm99analyze import Spm99AnalyzeImage
from .spm2analyze import Spm2AnalyzeImage
from .nifti1 import Nifti1Pair, Nifti1Image
from .minc1 import Minc1Image
from .freesurfer import MGHImage
from .parrec import PARRECImage
from .volumeutils import Recoder

# If we don't have scipy, then we cannot write SPM format files
try:
    import scipy.io
except ImportError:
    have_scipy = False
else:
    have_scipy = True

# mapping of names to classes and class functionality

class_map = {
    'analyze': {'class': AnalyzeImage, # Image class
                'ext': '.img', # characteristic image extension
                'has_affine': False, # class can store an affine
                'makeable': True, # empty image can be easily made in memory
                'rw': True}, # image can be written
    'spm99analyze': {'class': Spm99AnalyzeImage,
                     'ext': '.img',
                     'has_affine': True,
                     'makeable': True,
                     'rw': have_scipy},
    'spm2analyze': {'class': Spm2AnalyzeImage,
                    'ext': '.img',
                    'has_affine': True,
                    'makeable': True,
                    'rw': have_scipy},
    'nifti_pair': {'class': Nifti1Pair,
                   'ext': '.img',
                   'has_affine': True,
                   'makeable': True,
                   'rw': True},
    'nifti_single': {'class': Nifti1Image,
                     'ext': '.nii',
                     'has_affine': True,
                     'makeable': True,
                     'rw': True},
    'minc': {'class': Minc1Image,
             'ext': '.mnc',
             'has_affine': True,
             'makeable': True,
             'rw': False},
    'mgh':{'class': MGHImage,
           'ext': '.mgh',
           'has_affine': True,
           'makeable': True,
           'rw':True},
    'mgz':{'class': MGHImage,
           'ext': '.mgz',
           'has_affine': True,
           'makeable': True,
           'rw':True},
    'par':{'class': PARRECImage,
           'ext': '.par',
           'has_affine': True,
           'makeable': False,
           'rw' : False}}

# mapping of extensions to default image class names
ext_map = Recoder((
    ('nifti_single', '.nii'),
    ('nifti_pair', '.img', '.hdr'),
    ('minc', '.mnc'),
    ('mgh', '.mgh'),
    ('mgz', '.mgz'),
    ('par', '.par'),
))
