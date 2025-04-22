# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""GIfTI format IO

.. currentmodule:: nibabel.gifti

.. autosummary::
   :toctree: ../generated

   gifti
"""

from .gifti import (
    GiftiCoordSystem,
    GiftiDataArray,
    GiftiImage,
    GiftiLabel,
    GiftiLabelTable,
    GiftiMetaData,
    GiftiNVPairs,
)
