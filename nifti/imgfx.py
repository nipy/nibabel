#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Functions operating on images"""

__docformat__ = 'restructuredtext'


import numpy as N
import nifti.clib as ncl


def crop(nimg, bbox):
    """Crop an image.

    :Parameters:
      bbox: list(2-tuples)
        Each tuple has the (min,max) values for a particular image dimension.

    :Returns:
      A cropped image. The data is not shared with the original image, but
      is copied.
    """

    # build crop command
    # XXX: the following looks rather stupid -- cannot recall why I did this
    cmd = 'nimg.data.squeeze()['
    cmd += ','.join( [ ':'.join( [ str(i) for i in dim ] ) for dim in bbox ] )
    cmd += ']'

    # crop the image data array
    cropped = eval(cmd).copy()

    # return the cropped image with preserved header data
    return nimg.__class__(cropped, nimg.header)
