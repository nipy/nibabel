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
from .deprecated import deprecate_with_version


warnings.warn('The checkwarns module is deprecated and will be removed '
              'in nibabel v3.0', DeprecationWarning)


@deprecate_with_version('ErrorWarnings is deprecated; use nibabel.testing.error_warnings.',
                        since='2.1.0', until='3.0.0')
class ErrorWarnings(error_warnings):
    pass


@deprecate_with_version('IgnoreWarnings is deprecated; use nibabel.testing.suppress_warnings.',
                        since='2.1.0', until='3.0.0')
class IgnoreWarnings(suppress_warnings):
    pass
