#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Python bindings to the NIfTI data format


Package Organization
====================

The nifti package contains the following subpackages and modules:

.. packagetree::
   :style: UML

:author: `Michael Hanke <michael.hanke@gmail.com>`__
:requires: Python 2.4+
:version: 0.20070930.1
:see: `The PyNIfTI webpage <http://niftilib.sf.net/pynifti>`__
:see: `GIT Repository Browser <http://git.debian.org/?p=pkg-exppsy/pynifti.git>`__

:license: The MIT License
:copyright: |copy| 2006-2008 Michael Hanke <michael.hanke@gmail.com>

:newfield contributor: Contributor, Contributors (Alphabetical Order)
:contributor: `Yaroslav Halchenko <debian@onerussian.com>`__

.. |copy| unicode:: 0xA9 .. copyright sign
"""

__docformat__ = 'restructuredtext'


from nifti.niftiimage import NiftiImage
