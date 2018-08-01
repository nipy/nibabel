# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Geometric transforms

.. currentmodule:: nibabel.transform

.. autosummary::
   :toctree: ../generated

   transform
"""
from __future__ import absolute_import
from .base import ImageSpace, Affine
from .nonlinear import DeformationFieldTransform
