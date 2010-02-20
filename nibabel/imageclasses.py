''' Define supported image classes and names '''
from nibabel.analyze import AnalyzeImage
from nibabel.spm99analyze import Spm99AnalyzeImage
from nibabel.spm2analyze import Spm2AnalyzeImage
from nibabel.nifti1 import Nifti1Pair, Nifti1Image
from nibabel.minc import MincImage

# mapping of names to classes and class functionality
class_map = {
    'analyze': {'class': AnalyzeImage,
                'ext': '.img',
                'has_affine': False,
                'rw': True},
    'spm99analyze': {'class': Spm99AnalyzeImage,
                     'ext': '.img',
                     'has_affine': True,
                     'rw': True},
    'spm2analyze': {'class': Spm2AnalyzeImage,
                    'ext': '.img',
                    'has_affine': True,
                    'rw': True},
    'nifti_pair': {'class': Nifti1Pair,
                   'ext': '.img',
                    'has_affine': True,                   
                   'rw': True},
    'nifti_single': {'class': Nifti1Image,
                     'ext': '.nii',
                     'has_affine': True,
                     'rw': True},
    'minc': {'class': MincImage,
             'ext': '.mnc',
             'has_affine': True,
             'rw': False}}

