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


class ErrorWarnings(warnings.catch_warnings):
    """ Context manager to check for warnings as errors.  Usually used with
    ``assert_raises`` in the with block

    Examples
    --------
    >>> with ErrorWarnings():
    ...     try:
    ...         warnings.warn('Message', UserWarning)
    ...     except UserWarning:
    ...         print('I consider myself warned')
    I consider myself warned
    """
    filter = 'error'
    def __init__(self, record=True, module=None):
        super(ErrorWarnings, self).__init__(record=record, module=module)

    def __enter__(self):
        mgr = super(ErrorWarnings, self).__enter__()
        warnings.simplefilter(self.filter)
        return mgr


class IgnoreWarnings(ErrorWarnings):
    """ Context manager to ignore warnings

    Examples
    --------
    >>> with IgnoreWarnings():
    ...     warnings.warn('Message', UserWarning)

    (and you get no warning)
    """
    filter = 'ignore'
