import typing as ty

import numpy as np
import pytest

from nibabel import AnalyzeImage, Spm99AnalyzeImage, Spm2AnalyzeImage, Nifti1Image, Nifti2Image, MGHImage
from nibabel.spatialimages import SpatialImage

if ty.TYPE_CHECKING:
    from typing import reveal_type
else:

    def reveal_type(x: ty.Any) -> None:
        pass


@pytest.mark.mypy_testing
def test_affine_tracking() -> None:
    img_with_affine = SpatialImage(np.empty((5, 5, 5)), np.eye(4))
    img_without_affine = SpatialImage(np.empty((5, 5, 5)), None)

    reveal_type(img_with_affine)  # R: nibabel.spatialimages.SpatialImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    reveal_type(img_without_affine)  # R: nibabel.spatialimages.SpatialImage[None]


@pytest.mark.mypy_testing
def test_SpatialImageAPI() -> None:
    img = SpatialImage(np.empty((5, 5, 5)), np.eye(4))

    # Affine
    reveal_type(img.affine)  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]
    reveal_type(SpatialImage(np.empty((5, 5, 5)), None).affine)  # R: None

    # Data
    reveal_type(img.dataobj)  # R: nibabel.arrayproxy.ArrayLike
    reveal_type(img.get_fdata())  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]
    reveal_type(img.get_fdata(dtype=np.float32))  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[numpy._typing._nbit_base._32Bit]]]
    reveal_type(img.get_fdata(dtype=np.float64))  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]
    reveal_type(img.get_fdata(dtype=np.dtype(np.float32)))  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[numpy._typing._nbit_base._32Bit]]]
    reveal_type(img.get_fdata(dtype=np.dtype(np.float64)))  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]
    reveal_type(img.get_fdata(dtype=np.dtype("f4")))  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[numpy._typing._nbit_base._32Bit]]]
    reveal_type(img.get_fdata(dtype=np.dtype("f8")))  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]
    reveal_type(img.get_fdata(dtype="f4"))  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[numpy._typing._nbit_base._32Bit]]]
    reveal_type(img.get_fdata(dtype="f8"))  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]

    # Indirect header
    reveal_type(img.shape)  # R: builtins.tuple[builtins.int, ...]
    reveal_type(img.ndim)  # R: builtins.int

    # SpatialHeader fields
    reveal_type(img.header.get_data_dtype())  # R: numpy.dtype[Any]
    reveal_type(img.header.get_data_shape())  # R: builtins.tuple[builtins.int, ...]
    reveal_type(img.header.get_zooms())  # R: builtins.tuple[builtins.float, ...]


@pytest.mark.mypy_testing
def test_image_and_header_types() -> None:
    analyze_img = AnalyzeImage(np.empty((5, 5, 5)), np.eye(4))
    reveal_type(analyze_img)  # R: nibabel.analyze.AnalyzeImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    reveal_type(analyze_img.header)  # R: nibabel.analyze.AnalyzeHeader

    spm99_img = Spm99AnalyzeImage(np.empty((5, 5, 5)), np.eye(4))
    reveal_type(spm99_img)  # R: nibabel.spm99analyze.Spm99AnalyzeImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    reveal_type(spm99_img.header)  # R: nibabel.spm99analyze.Spm99AnalyzeHeader

    spm2_img = Spm2AnalyzeImage(np.empty((5, 5, 5)), np.eye(4))
    reveal_type(spm2_img)  # R: nibabel.spm2analyze.Spm2AnalyzeImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    reveal_type(spm2_img.header)  # R: nibabel.spm2analyze.Spm2AnalyzeHeader

    ni1_img = Nifti1Image(np.empty((5, 5, 5)), np.eye(4))
    reveal_type(ni1_img)  # R: nibabel.nifti1.Nifti1Image[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    reveal_type(ni1_img.header)  # R: nibabel.nifti1.Nifti1Header

    ni2_img = Nifti2Image(np.empty((5, 5, 5)), np.eye(4))
    reveal_type(ni2_img)  # R: nibabel.nifti2.Nifti2Image[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    reveal_type(ni2_img.header)  # R: nibabel.nifti2.Nifti2Header

    mgh_img = MGHImage(np.empty((5, 5, 5), dtype=np.float32), np.eye(4))
    reveal_type(mgh_img)  # R: nibabel.freesurfer.mghformat.MGHImage
    reveal_type(mgh_img.header)  # R: nibabel.freesurfer.mghformat.MGHHeader
