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
import h5py

from scipy import ndimage as ndi


class ImageSpace(object):
    '''Class to represent spaces of gridded data (images)'''
    __slots__ = ['_affine', '_shape', '_ndim', '_ndindex', '_coords', '_nvox',
                 '_inverse']

    def __init__(self, image):
        self._affine = image.affine
        self._shape = image.shape
        self._ndim = len(image.shape)
        self._nvox = np.prod(image.shape)  # Do not access data array
        self._ndindex = None
        self._coords = None
        self._inverse = np.linalg.inv(image.affine)
        if self._ndim not in [2, 3]:
            raise ValueError('Invalid image space (%d-D)' % self._ndim)

    @property
    def affine(self):
        return self._affine

    @property
    def inverse(self):
        return self._inverse

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self._ndim

    @property
    def nvox(self):
        return self._nvox

    @property
    def ndindex(self):
        if self._ndindex is None:
            indexes = tuple([np.arange(s) for s in self._shape])
            self._ndindex = np.array(np.meshgrid(
                *indexes, indexing='ij')).reshape(self._ndim, self._nvox)
        return self._ndindex

    @property
    def ndcoords(self):
        if self._coords is None:
            self._coords = np.tensordot(
                self._affine,
                np.vstack((self.ndindex, np.ones((1, self._nvox)))),
                axes=1
            )[:3, ...]
        return self._coords

    def map_voxels(self, coordinates):
        coordinates = np.array(coordinates)
        ncoords = coordinates.shape[-1]
        coordinates = np.vstack((coordinates, np.ones((1, ncoords))))

        # Back to grid coordinates
        return np.tensordot(np.linalg.inv(self._affine),
                            coordinates, axes=1)[:3, ...]

    def __eq__(self, other):
        try:
            return np.allclose(self.affine, other.affine) and self.shape == other.shape
        except AttributeError:
            pass
        return False


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

    @property
    def ndim(self):
        return self.reference.ndim

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

    def _to_hdf5(self, x5_root):
        '''Serialize this object into the x5 file format'''
        raise NotImplementedError

    def to_filename(self, filename):
        '''Store the transform in BIDS-Transforms HDF5 file format (.x5).
        '''
        with h5py.File(filename, 'w') as out_file:
            out_file['Format'] = 'X5'
            out_file['Version'] = np.uint16(1)
            root = out_file.create_group('/0')
            self._to_hdf5(root)

        return filename
