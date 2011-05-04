# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Contexts for *with* statement allowing checks for warnings

When we give up 2.5 compatibility we can use python's own
``tests.test_support.check_warnings``

'''
from __future__ import with_statement

import warnings


class ErrorWarnings(object):
    """ Context manager to check for warnings as errors.  Usually used with
    ``assert_raises`` in the with block

    Examples
    --------
    >>> with ErrorWarnings():
    ...     try:
    ...         warnings.warn('Message', UserWarning)
    ...     except UserWarning:
    ...         print 'I consider myself warned'
    I consider myself warned

    Notes
    -----
    The manager will raise a RuntimeError if another warning filter gets put on
    top of the one it has just added.
    """
    def __init__(self):
        self.added = None

    def __enter__(self):
        warnings.simplefilter('error')
        self.added = warnings.filters[0]

    def __exit__(self, exc, value, tb):
        if warnings.filters[0] != self.added:
            raise RuntimeError('Somone has done something to the filters')
        warnings.filters.pop(0)
        return False # allow any exceptions to propagate


class IgnoreWarnings(ErrorWarnings):
    """ Context manager to ignore warnings

    Examples
    --------
    >>> with IgnoreWarnings():
    ...     warnings.warn('Message', UserWarning)

    (and you get no warning)

    Notes
    -----
    The manager will raise a RuntimeError if another warning filter gets put on
    top of the one it has just added.
    """

    def __enter__(self):
        warnings.simplefilter('ignore')
        self.added = warnings.filters[0]
