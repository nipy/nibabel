# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import os
from typing import Any

# module imports
from nibabel import analyze as ana
from nibabel import ecat, imagestats, mriutils, orientations, streamlines, viewers
from nibabel import nifti1 as ni1
from nibabel import spm2analyze as spm2
from nibabel import spm99analyze as spm99

# object imports
from nibabel.analyze import AnalyzeHeader, AnalyzeImage
from nibabel.arrayproxy import is_proxy
from nibabel.cifti2 import Cifti2Header, Cifti2Image
from nibabel.fileholders import FileHolder, FileHolderError
from nibabel.freesurfer import MGHImage
from nibabel.funcs import as_closest_canonical, concat_images, four_to_three, squeeze_image
from nibabel.gifti import GiftiImage
from nibabel.imageclasses import all_image_classes
from nibabel.info import long_description as __doc__
from nibabel.loadsave import load, save
from nibabel.minc1 import Minc1Image
from nibabel.minc2 import Minc2Image
from nibabel.nifti1 import Nifti1Header, Nifti1Image, Nifti1Pair
from nibabel.nifti2 import Nifti2Header, Nifti2Image, Nifti2Pair
from nibabel.orientations import (
    OrientationError,
    aff2axcodes,
    apply_orientation,
    flip_axis,
    io_orientation,
)
from nibabel.pkg_info import __version__
from nibabel.pkg_info import get_pkg_info as _get_pkg_info
from nibabel.spm2analyze import Spm2AnalyzeHeader, Spm2AnalyzeImage
from nibabel.spm99analyze import Spm99AnalyzeHeader, Spm99AnalyzeImage

def get_info() -> dict[str, str]: ...
def test(
    label: Any = None,
    verbose: int = 1,
    extra_argv: list[Any] | None = None,
    doctests: bool = False,
    coverage: bool = False,
    raise_warnings: Any = None,
    timer: Any = False,
) -> int: ...
def bench(label: Any = None, verbose: int = 1, extra_argv: list[Any] | None = None) -> int: ...

__all__ = [
    'AnalyzeHeader',
    'AnalyzeImage',
    'Cifti2Header',
    'Cifti2Image',
    'FileHolder',
    'FileHolderError',
    'GiftiImage',
    'MGHImage',
    'Minc1Image',
    'Minc2Image',
    'Nifti1Header',
    'Nifti1Image',
    'Nifti1Pair',
    'Nifti2Header',
    'Nifti2Image',
    'Nifti2Pair',
    'OrientationError',
    'Spm2AnalyzeHeader',
    'Spm2AnalyzeImage',
    'Spm99AnalyzeHeader',
    'Spm99AnalyzeImage',
    '__doc__',
    '__version__',
    '_get_pkg_info',
    'aff2axcodes',
    'all_image_classes',
    'ana',
    'apply_orientation',
    'as_closest_canonical',
    'bench',
    'concat_images',
    'ecat',
    'flip_axis',
    'four_to_three',
    'get_info',
    'imagestats',
    'io_orientation',
    'is_proxy',
    'load',
    'mriutils',
    'ni1',
    'orientations',
    'os',
    'save',
    'spm2',
    'spm99',
    'squeeze_image',
    'streamlines',
    'test',
    'viewers',
]
