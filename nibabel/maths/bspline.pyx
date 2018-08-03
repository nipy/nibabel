# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Cython math extension '''
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs


cdef double c_cubic(double x) nogil:
    cdef:
        double x_t = fabs(x)

    if x_t >= 2.0:
        return(0.0)
    if x_t <= 1.0:
        return(2.0 / 3.0 - x_t**2 + 0.5 * x_t**3)
    elif x_t <= 2.0:
        return((2 - x_t)**3 / 6.0)


def cubic(double x):
    """
    Evaluate the univariate cubic bspline at x

    Pure python implementation: ::

        def bspl(x):
            if x >= 2.0:
                return 0.0
            if x <= 1.0:
                return 2.0 / 3.0 - x**2 + 0.5 * x**3
            elif x <= 2.0:
                return (2 - x)**3 / 6.0
    """
    return(c_cubic(x))
