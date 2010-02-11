''' Define supported image classes and names '''
from nibabel.analyze import AnalyzeImage
from nibabel.spm99analyze import Spm99AnalyzeImage
from nibabel.spm2analyze import Spm2AnalyzeImage
from nibabel.nifti1 import Nifti1Pair, Nifti1Image
from nibabel.minc import MincImage

image_classes = {
    'analyze': AnalyzeImage,
    'spm99analyze': Spm99AnalyzeImage,
    'spm2analyze': Spm2AnalyzeImage,
    'nifti_pair': Nifti1Pair,
    'nifti_single': Nifti1Image,
    'minc': MincImage}

