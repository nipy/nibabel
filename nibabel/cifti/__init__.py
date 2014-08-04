# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""CIfTI format IO

.. currentmodule:: nibabel.cifti

.. autosummary::
   :toctree: ../generated

   ciftiio
   cifti
"""

from .parse_cifti_fast import create_cifti_image
from .cifti import (CiftiMetaData, CiftiHeader, CiftiImage, CiftiLabel,
                    CiftiLabelTable, CiftiNVPair, CiftiVertexIndices,
                    CiftiVoxelIndicesIJK, CiftiBrainModel, CiftiMatrix,
                    CiftiMatrixIndicesMap, CiftiNamedMap, CiftiParcel,
                    CiftiSurface, CiftiTransformationMatrixVoxelIndicesIJKtoXYZ,
                    CiftiVertices, CiftiVolume, CIFTI_BrainStructures,
                    CIFTI_MODEL_TYPES, load, save)

