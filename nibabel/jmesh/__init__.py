# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""JSON and BJData based JMesh format IO

.. currentmodule:: nibabel.jmesh

.. autosummary::
   :toctree: ../generated

   jmesh
"""

from .jmesh import load, save, JMesh, default_header
