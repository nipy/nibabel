#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Only here for backward compatibility."""

__docformat__ = 'restructuredtext'

from warnings import warn

warn("This module has been renamed to 'nifti.format'. This redirect will be removed with PyNIfTI 1.0.", DeprecationWarning)



from nifti.format import *
