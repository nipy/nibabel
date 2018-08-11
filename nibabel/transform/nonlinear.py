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
# from gridbspline.interpolate import BsplineNDInterpolator

from .base import ImageSpace, TransformBase
from ..funcs import four_to_three


class DeformationFieldTransform(TransformBase):
    '''Represents a dense field of displacements (one vector per voxel)'''
    __slots__ = ['_field', '_moving', '_moving_space']
    __s = (slice(None), )

    def __init__(self, field, reference=None):
        '''
        Create a dense deformation field transform
        '''
        super(DeformationFieldTransform, self).__init__()
        self._field = field.get_data()

        ndim = self._field.ndim - 1
        if self._field.shape[:-1] != ndim:
            raise ValueError(
                'Number of components of the deformation field does '
                'not match the number of dimensions')

        self._moving = None  # Where each voxel maps to
        self._moving_space = None  # Input space cache

        # By definition, a free deformation field has a
        # displacement vector per voxel in output (reference)
        # space
        if reference is None:
            reference = four_to_three(field)[0]
        elif reference.shape[:ndim] != field.shape[:ndim]:
            raise ValueError(
                'Reference ({}) and field ({}) must have the same '
                'grid.'.format(
                    _pprint(reference.shape[:ndim]),
                    _pprint(field.shape[:ndim])))

        self.reference = reference

    def _cache_moving(self, moving):
        # Check whether input (moving) space is cached
        moving_space = ImageSpace(moving)
        if self._moving_space == moving_space:
            return

        # Generate grid of pixel indexes (ijk)
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

        # Calculate physical coords of all voxels (xyz)
        flatxyz = np.tensordot(
            self.reference.affine,
            np.vstack((flatgrid, np.ones((1, flatgrid.shape[1])))),
            axes=1
        )

        # Add field
        newxyz = flatxyz + np.vstack((
            np.moveaxis(self._field, -1, 0).reshape(ndim, -1),
            np.zeros((1, flatgrid.shape[1]))))

        # Back to grid coordinates
        newijk = np.tensordot(np.linalg.inv(moving.affine),
                              newxyz, axes=1)

        # Reshape as grid
        self._moving = np.moveaxis(
            newijk[0:3, :].reshape((ndim, ) + self._field.shape[:-1]),
            0, -1)

        self._moving_space = moving_space

    def resample(self, moving, order=3, mode='constant', cval=0.0, prefilter=True,
                 output_dtype=None):
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
        self._cache_moving(moving)
        return super(DeformationFieldTransform, self).resample(
            moving, order=order, mode=mode, cval=cval, prefilter=prefilter)

    def map_voxel(self, index, moving=None):
        return tuple(self._moving[index + self.__s])

    def map_coordinates(self, coordinates, order=3, mode='mirror', cval=0.0,
                        prefilter=True):
        coordinates = np.array(coordinates)
        # Extract shapes and dimensions, then flatten
        ndim = coordinates.shape[-1]
        output_shape = coordinates.shape[:-1]
        flatcoord = np.moveaxis(coordinates, -1, 0).reshape(ndim, -1)

        # Convert coordinates to voxel indices
        ijk = np.tensordot(
            np.linalg.inv(self.reference.affine),
            np.vstack((flatcoord, np.ones((1, flatcoord.shape[1])))),
            axes=1)
        deltas = ndi.map_coordinates(
            self._field,
            ijk,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=prefilter)

        deltas = np.moveaxis(deltas[0:3, :].reshape((ndim, ) + output_shape),
                             0, -1)

        return coordinates + deltas


class BSplineFieldTransform(TransformBase):
    __slots__ = ['_coeffs', '_knots', '_refknots']

    def __init__(self, reference, coefficients, order=3):
        '''Create a smooth deformation field using B-Spline basis'''
        super(BSplineFieldTransform, self).__init__()
        self.reference = reference

        if coefficients.shape[-1] != self.ndim:
            raise ValueError(
                'Number of components of the coefficients does '
                'not match the number of dimensions')

        self._coeffs = coefficients.get_data().reshape(-1, self.ndim)
        self._knots = ImageSpace(four_to_three(coefficients)[0])

        # Calculate the voxel coordinates of the reference image
        # in the B-spline basis space.
        self._refknots = np.tensordot(
            np.linalg.inv(self.knots.affine),
            np.vstack((self.reference.ndcoords, np.ones((1, self.reference.nvox)))),
            axes=1)[..., :3].reshape(self.reference.shape + (self._knots.ndim, ))


    def map_voxel(self, index, moving=None):

        indexes = []
        offset = 0.0 if self._order & 1 else 0.5
        for dim in range(self.ndim):
            first = int(np.floor(xi[dim] + offset) - self._order // 2)
            indexes.append(list(range(first, first + self._order + 1)))

        ndindex = np.moveaxis(
            np.array(np.meshgrid(*indexes, indexing='ij')), 0, -1).reshape(
            -1, self.ndim)

        vbspl = np.vectorize(cubic)
        weights = np.prod(vbspl(ndindex - xi), axis=-1)
        ndindex = [tuple(v) for v in ndindex]

        zero = np.zeros(self.ndim)
        shape = np.array(self.shape)
        coeffs = []
        for ijk in ndindex:
            offbounds = (zero > ijk) | (shape <= ijk)
            if np.any(offbounds):
                # Deal with offbounds samples
                if self._off_bounds == 'constant':
                    coeffs.append([self._fill_value] * self.ncomp)
                    continue
                ijk = np.array(ijk, dtype=int)
                ijk[ijk < 0] *= -1
                ijk[ijk >= shape] = (2 * shape[ijk >= shape] - ijk[ijk >= shape] - 1).astype(int)
                ijk = tuple(ijk.tolist())

            coeffs.append(self._coeffs[ijk])
        return weights.dot(np.array(coeffs, dtype=float))


    # def resample(self, moving, order=3, mode='constant', cval=0.0, prefilter=True,
    #              output_dtype=None):
    #     '''

    #     Examples
    #     --------

    #     >>> import numpy as np
    #     >>> import nibabel as nb
    #     >>> ref = nb.load('t1_weighted.nii.gz')
    #     >>> coeffs = np.zeros((6, 6, 6, 3))
    #     >>> coeffs[2, 2, 2, ...] = [10.0, -20.0, 0]
    #     >>> aff = ref.affine
    #     >>> aff[:3, :3] = aff.dot(np.eye(3) * np.array(ref.header.get_zooms()[:3] / 6.0)
    #     >>> coeffsimg = nb.Nifti1Image(coeffs, ref.affine, ref.header)
    #     >>> xfm = nb.transform.BSplineFieldTransform(ref, coeffsimg)
    #     >>> new = xfm.resample(ref)
    #     >>> new.to_filename('deffield.nii.gz')

    #     '''

    #     return super(BSplineFieldTransform, self).resample(
    #         moving, order=order, mode=mode, cval=cval, prefilter=prefilter)


def _pprint(inlist):
    return 'x'.join(['%d' % s for s in inlist])
