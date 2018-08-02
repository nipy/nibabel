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
        self._field = field.get_data()
        self._moving = None

    def _cache_moving(self, moving):
        # Cache the new indexes
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
            np.vstack((flatgrid, np.ones((1, flatgrid.shape[1])))),
            axes=1
        )

        newxyz = flatxyz + np.vstack((self._field.reshape(ndim , -1),
                                     np.zeros((1, flatgrid.shape[1]))))
        newijk = np.tensordot(np.linalg.inv(moving.affine),
                              newxyz, axes=1)

        self._moving = np.moveaxis(
            newijk[0:3, :].reshape((ndim, ) + self._field.shape[:-1]),
            0, -1)

    def resample(self, moving, order=3, mode='constant', cval=0.0, prefilter=True,
                 output_dtype=None):
        self._cache_moving(moving)
        return super(DeformationFieldTransform, self).resample(
            moving, order=order, mode=mode, cval=cval, prefilter=prefilter)

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
        return tuple(self._moving[index + (slice(None), )])

