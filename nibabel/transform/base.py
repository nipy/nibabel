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
from scipy import ndimage as ndi


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

    def resample(self, moving, order=3, mode='constant', cval=0.0, prefilter=True,
                 output_dtype=None):
        '''Resample the moving image in reference space

        Parameters
        ----------

        moving : `spatialimage`
            The image object containing the data to be resampled in reference
            space
        order : int, optional
            The order of the spline interpolation, default is 3.
            The order has to be in the range 0-5.
        mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
            Determines how the input image is extended when the resamplings overflows
            a border. Default is 'constant'.
        cval : float, optional
            Constant value for ``mode='constant'``. Default is 0.0.
        prefilter: bool, optional
            Determines if the moving image's data array is prefiltered with
            a spline filter before interpolation. The default is ``True``,
            which will create a temporary *float64* array of filtered values
            if *order > 1*. If setting this to ``False``, the output will be
            slightly blurred if *order > 1*, unless the input is prefiltered,
            i.e. it is the result of calling the spline filter on the original
            input.

        Returns
        -------

        moved_image : `spatialimage`
            The moving imaged after resampling to reference space.

        '''

        if output_dtype is None:
            output_dtype = moving.header.get_data_dtype()

        moving_data = moving.get_data()
        moved = ndi.geometric_transform(
            moving_data,
            mapping=self.map_voxel,
            output_shape=self.reference.shape,
            output=output_dtype,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=prefilter,
            extra_keywords={'moving': moving},
        )

        moved_image = moving.__class__(moved, self.reference.affine, moving.header)
        moved_image.header.set_data_dtype(output_dtype)
        return moved_image

    def map_point(self, coords):
        '''Find the coordinates in moving space corresponding to the
        input reference coordinates'''
        raise NotImplementedError

    def map_voxel(self, index, moving=None):
        '''Find the voxel indices in the moving image corresponding to the
        input reference voxel'''
        raise NotImplementedError


class Affine(TransformBase):
    '''Represents linear transforms on image data'''
    __slots__ = ['_matrix']

    def __init__(self, matrix):
        '''Initialize a transform

        Parameters
        ----------

        matrix : ndarray
            The inverse coordinate transformation matrix **in physical
            coordinates**, mapping coordinates from *reference* space
            into *moving* space.
            This matrix should be provided in homogeneous coordinates.

        Examples
        --------

        >>> xfm = Affine([[1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        >>> xfm.matrix  # doctest: +NORMALIZE_WHITESPACE
        array([[1, 0, 0, 4],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

        '''
        self._matrix = np.array(matrix)
        assert self._matrix.ndim == 2, 'affine matrix should be 2D'
        assert self._matrix.shape[0] == self._matrix.shape[1], 'affine matrix is not square'
        super(Affine, self).__init__()

    @property
    def matrix(self):
        return self._matrix

    def resample(self, moving, order=3, mode='constant', cval=0.0, prefilter=True,
                 output_dtype=None):
        '''Resample the moving image in reference space

        Parameters
        ----------

        moving : `spatialimage`
            The image object containing the data to be resampled in reference
            space
        order : int, optional
            The order of the spline interpolation, default is 3.
            The order has to be in the range 0-5.
        mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
            Determines how the input image is extended when the resamplings overflows
            a border. Default is 'constant'.
        cval : float, optional
            Constant value for ``mode='constant'``. Default is 0.0.
        prefilter: bool, optional
            Determines if the moving image's data array is prefiltered with
            a spline filter before interpolation. The default is ``True``,
            which will create a temporary *float64* array of filtered values
            if *order > 1*. If setting this to ``False``, the output will be
            slightly blurred if *order > 1*, unless the input is prefiltered,
            i.e. it is the result of calling the spline filter on the original
            input.

        Returns
        -------

        moved_image : `spatialimage`
            The moving imaged after resampling to reference space.


        Examples
        --------

        >>> import nibabel as nib
        >>> xfm = Affine([[1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        >>> ref = nib.load('image.nii.gz')
        >>> xfm.reference = ref
        >>> xfm.resample(ref, order=0)

        '''
        if output_dtype is None:
            output_dtype = moving.header.get_data_dtype()

        try:
            reference = self.reference
        except ValueError:
            print('Warning: no reference space defined, using moving as reference')
            reference = moving

        # Compose an index to index affine matrix
        matrix = reference.affine.dot(self._matrix.dot(np.linalg.inv(moving.affine)))
        mdata = moving.get_data()
        moved = ndi.affine_transform(
            mdata, matrix=matrix[:mdata.ndim, :mdata.ndim],
            offset=matrix[:mdata.ndim, mdata.ndim],
            output_shape=reference.shape, order=order, mode=mode,
            cval=cval, prefilter=prefilter)

        moved_image = moving.__class__(moved, reference.affine, moving.header)
        moved_image.header.set_data_dtype(output_dtype)

        return moved_image

    def map_point(self, coords, forward=True):
        coords = np.array(coords)
        if coords.shape[0] == self._matrix.shape[0] - 1:
            coords = np.append(coords, [1])
        affine = self._matrix if forward else np.linalg.inv(self._matrix)
        return affine.dot(coords)[:-1]

    def map_voxel(self, index, moving=None):
        try:
            reference = self.reference
        except ValueError:
            print('Warning: no reference space defined, using moving as reference')
            reference = moving
        else:
            if moving is None:
                moving = reference
        finally:
            if reference is None:
                raise ValueError('Reference and moving spaces are both undefined')

        index = np.array(index)
        if index.shape[0] == self._matrix.shape[0] - 1:
            index = np.append(index, [1])

        matrix = reference.affine.dot(self._matrix.dot(np.linalg.inv(moving.affine)))
        return matrix.dot(index)[:-1]
