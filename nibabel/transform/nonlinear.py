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
    __slots__ = ['_field']

    def __init__(self, field):
        self.reference = four_to_three(field)[0]
        # Set the field components in front
        self._field = field.get_data()

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
        vox2ras = self.reference.affine
        ras2vox = np.linalg.inv(vox2ras if moving is None
                                else moving.affine)

        index = np.array(index)
        if index.shape[0] == vox2ras.shape[0] - 1:
            index = np.append(index, [1.0])

        idx = tuple([int(i) for i in index[:-1]])
        point = vox2ras.dot(index)[:-1]
        delta = np.squeeze(self._field[idx + (slice(None), )])
        newindex = ras2vox.dot(np.append(point + delta, [1]))[:-1]
        return tuple(newindex)
