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


class ImageSpace(object):
    '''
    Abstract class to represent spaces of gridded data (images)
    '''
    __slots__ = ['_affine', '_shape']

    def __init__(self, image):
        self._affine = image.affine
        self._shape = image.shape

    @property
    def affine(self):
        return self._affine

    @property
    def shape(self):
        return self._shape


class TransformBase(object):
    '''
    Abstract image class to represent transforms
    '''
    __slots__ = ['_reference']

    def __init__(self):
        self._reference = None

    @property
    def reference(self):
        '''A reference space where data will be resampled onto'''
        if self._reference is None:
            raise ValueError('Reference space not set')
        return self._reference

    @reference.setter
    def reference(self, image):
        self._reference = ImageSpace(image)

    def resample(self, moving, order=3, dtype=None):
        '''A virtual method to resample the moving image in the reference space'''
        raise NotImplementedError

    def transform(self, coords, vox=False):
        '''Find the coordinates in moving space corresponding to the
        input reference coordinates'''
        raise NotImplementedError


class Affine(TransformBase):
    '''Linear transforms'''
    __slots__ = ['_affine']

    def __init__(self, affine):
        self._affine = np.array(affine)
        assert self._affine.ndim == 2, 'affine matrix should be 2D'
        assert self._affine.shape[0] == self._affine.shape[1], 'affine matrix is not square'
        super(Affine, self).__init__()

    def resample(self, moving, order=3, dtype=None):

        if dtype is None:
            dtype = moving.get_data_dtype()

        # Compose an index to index affine matrix
        xfm = self.reference.affine.dot(self._affine.dot(np.linalg.inv(moving.affine)))

    def transform(self, coords):
        return self._affine.dot(coords)[:-1]



