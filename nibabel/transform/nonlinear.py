# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Common interface for transforms '''
from __future__ import division, print_function, absolute_import
import numpy as np

from .base import TransformBase
from ..funcs import four_to_three


class DeformationFieldTransform(TransformBase):
    '''Represents linear transforms on image data'''
    __slots__ = ['_field', '_moving']

    def __init__(self, field):
        super(DeformationFieldTransform, self).__init__()
        self.reference = four_to_three(field)[0]
        # Cache the new indexes
        self._field = field.get_data()
        ndim = self._field.ndim - 1
        if ndim == 2:
            grid = np.meshgrid(
                np.arange(self._field.shape[0]),
                np.arange(self._field.shape[1]),
                indexing='ij')
        elif ndim == 3:
            grid = np.meshgrid(
                np.arange(self._field.shape[0]),
                np.arange(self._field.shape[1]),
                np.arange(self._field.shape[2]),
                indexing='ij')
        else:
            raise ValueError('Wrong dimensions (%d)' % ndim)

        grid = np.array(grid)
        flatgrid = grid.reshape(ndim, -1)
        flatxyz = np.tensordot(
            self.reference.affine,
            np.vstack((flatgrid, np.ones((1, ndim)))),
            axes=1
        )

        delta = np.moveaxis(
            flatxyz[0:3, :].reshape((ndim, ) + self._field.shape[:-1]),
            0, -1)
        self._moving = self._field + delta

    def map_voxel(self, index, moving=None):
        '''

        Examples
        --------

        >>> import numpy as np
        >>> import nibabel as nb
        >>> ref = nb.load('t1_weighted.nii.gz')
        >>> field = np.zeros(tuple(list(ref.shape) + [3]))
        >>> field[..., 0] = 4.0
        >>> fieldimg = nb.Nifti1Image(field, ref.affine, ref.header)
        >>> xfm = nb.transform.DeformationFieldTransform(fieldimg)
        >>> new = xfm.resample(ref)
        >>> new.to_filename('deffield.nii.gz')

        '''
        ras2vox = np.linalg.inv(
            self.reference.affine if moving is None
            else moving.affine)

        point = self._moving[index + (slice(None), )]
        newindex = ras2vox.dot(np.append(point, [1]))[:-1]
        return tuple(newindex)
