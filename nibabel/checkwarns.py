# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Contexts for *with* statement allowing checks for warnings
'''
from __future__ import division, print_function

import warnings

from .testing import (error_warnings, suppress_warnings)


warnings.warn('The checkwarns module is deprecated and will be removed '
              'in nibabel v3.0', FutureWarning)


class ErrorWarnings(error_warnings):

    def __init__(self, *args, **kwargs):
        warnings.warn('ErrorWarnings is deprecated and will be removed in '
                      'nibabel v3.0; use nibabel.testing.error_warnings.',
                      FutureWarning)
        super(ErrorWarnings, self).__init__(*args, **kwargs)


class IgnoreWarnings(suppress_warnings):

    def __init__(self, *args, **kwargs):
        warnings.warn('IgnoreWarnings is deprecated and will be removed in '
                      'nibabel v3.0; use nibabel.testing.suppress_warnings.',
                      FutureWarning)
        super(IgnoreWarnings, self).__init__(*args, **kwargs)
