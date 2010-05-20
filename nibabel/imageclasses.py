''' Define supported image classes and names '''
from nibabel.analyze import AnalyzeImage
from nibabel.spm99analyze import Spm99AnalyzeImage
from nibabel.spm2analyze import Spm2AnalyzeImage
from nibabel.nifti1 import Nifti1Pair, Nifti1Image
from nibabel.minc import MincImage
from nibabel.volumeutils import Recoder

# If we don't have scipy, then we cannot write SPM format files
try:
    import scipy.io
except ImportError:
    have_scipy = False
else:
    have_scipy = True

# mapping of names to classes and class functionality
class_map = {
    'analyze': {'class': AnalyzeImage,
                'ext': '.img',
                'has_affine': False,
                'rw': True},
    'spm99analyze': {'class': Spm99AnalyzeImage,
                     'ext': '.img',
                     'has_affine': True,
                     'rw': have_scipy},
    'spm2analyze': {'class': Spm2AnalyzeImage,
                    'ext': '.img',
                    'has_affine': True,
                    'rw': have_scipy},
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


# mapping of extensions to default image class names
ext_map = Recoder((
    ('nifti_single', '.nii'),
    ('nifti_pair', '.img', '.hdr'),
    ('minc', '.mnc')))
