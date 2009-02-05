#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Functions operating on images"""

# NOTE: This is for functions which exclusively use the public interface of the
# NiftiImage class (i.e. convenience stuff that anybody could do). Moreover,
# these functions have to run without having to import 'nifti.image' itself,
# so they can be assigned to the NiftiImage class itself, as additional methods.


__docformat__ = 'restructuredtext'


import numpy as N


def getBoundingBox(nim):
    """Get the bounding box an image.

    The bounding box is the smallest box covering all non-zero elements.

    :Returns:
      tuple(2-tuples) | None
        It returns as many (min, max) tuples as there are image dimensions. The
        order of dimensions is identical to that in the data array. `None` is
        returned of the images does not contain non-zero elements.

    Examples:

      >>> from nifti import NiftiImage
      >>> nim = NiftiImage(N.zeros((12, 24, 32)))
      >>> nim.bbox is None
      True

      >>> nim.data[3,10,13] = 1
      >>> nim.data[6,20,26] = 1
      >>> nim.bbox
      ((3, 6), (10, 20), (13, 26))

      >>> nim.crop()
      >>> nim.data.shape
      (4, 11, 14)
      >>> nim.bbox
      ((0, 3), (0, 10), (0, 13))

    .. seealso::
      :attr:`nifti.image.NiftiImage.bbox`,
      :func:`nifti.imgfx.crop`
    """
    nz = nim.data.squeeze().nonzero()

    bbox = []

    for dim in nz:
        # check if there are nonzero elements at all
        if not len(dim):
            return None
        bbox.append((dim.min(), dim.max()))

    return tuple(bbox)


def crop(nim, bbox=None):
    """Crop an image.

    :Parameters:
      bbox: list(2-tuples) | None
        Each tuple has the (min,max) values for a particular image dimension.
        If `None`, the images actual bounding box is used for cropping.

    .. seealso::
      :attr:`nifti.image.NiftiImage.bbox`,
      :func:`nifti.imgfx.getBoundingBox`
    """

    if bbox is None:
        bbox = getBoundingBox(nim)

        # if image has no non.zero elements do nothing
        if bbox is None:
            # XXX: or raise something?
            return

    # build crop command
    # XXX: the following looks rather stupid -- cannot recall why I did this
    cmd = 'nim.data.squeeze()['
    cmd += ','.join(['%i:%i' % (dim[0], dim[1] + 1) for dim in bbox ])
    cmd += ']'

    # crop the image data array and assign it to the array
    nim.data = eval(cmd)
