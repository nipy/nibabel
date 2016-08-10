# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
from __future__ import division, print_function, absolute_import

from os.path import join as pjoin, dirname
import sys
from distutils.version import LooseVersion

import numpy as np

import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data

CIFTI2_DATA = pjoin(get_nibabel_data(), 'nitest-cifti2')



from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_raises)

DATA_FILE1 = pjoin(CIFTI2_DATA, '')
DATA_FILE2 = pjoin(CIFTI2_DATA,
                   'Conte69.MyelinAndCorrThickness.32k_fs_LR.dscalar.nii')
DATA_FILE3 = pjoin(CIFTI2_DATA,
                   'Conte69.MyelinAndCorrThickness.32k_fs_LR.dtseries.nii')
DATA_FILE4 = pjoin(CIFTI2_DATA,
                   'Conte69.MyelinAndCorrThickness.32k_fs_LR.ptseries.nii')
DATA_FILE5 = pjoin(CIFTI2_DATA,
                   'Conte69.parcellations_VGD11b.32k_fs_LR.dlabel.nii')
DATA_FILE6 = pjoin(CIFTI2_DATA, 'ones.dscalar.nii')
datafiles = [DATA_FILE2, DATA_FILE3, DATA_FILE4, DATA_FILE5, DATA_FILE6]


@needs_nibabel_data('nitest-cifti2')
def test_read_internal():
    img2 = ci.load(DATA_FILE6)
    assert_true(isinstance(img2.header, ci.Cifti2Header))
    assert_equal(img2.shape, (1, 91282))

@needs_nibabel_data('nitest-cifti2')
def test_read():
    img2 = nib.load(DATA_FILE6)
    assert_true(isinstance(img2.header, ci.Cifti2Header))
    assert_equal(img2.shape, (1, 91282))

@needs_nibabel_data('nitest-cifti2')
def test_version():
    for i, dat in enumerate(datafiles):
        img = nib.load(dat)
        assert_equal(LooseVersion(img.header.version), LooseVersion('2'))

@needs_nibabel_data('nitest-cifti2')
def test_readwritedata():
    with InTemporaryDirectory():
        for name in datafiles:
            img = ci.load(name)
            nib.save(img, 'test.nii')
            img2 = ci.load('test.nii')
            assert_equal(len(img.header.matrix),
                         len(img2.header.matrix))
            # Order should be preserved in load/save
            for mim1, mim2 in zip(img.header.matrix,
                                  img2.header.matrix):
                named_maps1 = [m_ for m_ in mim1 if isinstance(m_, ci.Cifti2NamedMap)]
                named_maps2 = [m_ for m_ in mim2 if isinstance(m_, ci.Cifti2NamedMap)]
                assert_equal(len(named_maps1), len(named_maps2))
                for map1, map2 in zip(named_maps1, named_maps2):
                    assert_equal(map1.map_name, map2.map_name)
                    if map1.label_table is None:
                        assert_true(map2.label_table is None)
                    else:
                        assert_equal(len(map1.label_table),
                                     len(map2.label_table))
            assert_array_almost_equal(img.data, img2.data)


@needs_nibabel_data('nitest-cifti2')
def test_nibabel_readwritedata():
    with InTemporaryDirectory():
        for name in datafiles:
            img = nib.load(name)
            nib.save(img, 'test.nii')
            img2 = nib.load('test.nii')
            assert_equal(len(img.header.matrix),
                         len(img2.header.matrix))
            # Order should be preserved in load/save
            for mim1, mim2 in zip(img.header.matrix,
                                  img2.header.matrix):
                named_maps1 = [m_ for m_ in mim1 if isinstance(m_, ci.Cifti2NamedMap)]
                named_maps2 = [m_ for m_ in mim2 if isinstance(m_, ci.Cifti2NamedMap)]
                assert_equal(len(named_maps1), len(named_maps2))
                for map1, map2 in zip(named_maps1, named_maps2):
                    assert_equal(map1.map_name, map2.map_name)
                    if map1.label_table is None:
                        assert_true(map2.label_table is None)
                    else:
                        assert_equal(len(map1.label_table),
                                     len(map2.label_table))
            assert_array_almost_equal(img.data, img2.data)


@needs_nibabel_data('nitest-cifti2')
def test_cifti2types():
    """Check that we instantiate Cifti2 classes correctly, and that our
    test files exercise all classes"""
    counter = {ci.Cifti2LabelTable: 0,
               ci.Cifti2Label: 0,
               ci.Cifti2NamedMap: 0,
               ci.Cifti2Surface: 0,
               ci.Cifti2VoxelIndicesIJK: 0,
               ci.Cifti2Vertices: 0,
               ci.Cifti2Parcel: 0,
               ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ: 0,
               ci.Cifti2Volume: 0,
               ci.Cifti2VertexIndices: 0,
               ci.Cifti2BrainModel: 0,
               ci.Cifti2MatrixIndicesMap: 0,
               }

    for name in datafiles:
        hdr = ci.load(name).header
        # Matrix and MetaData aren't conditional, so don't bother counting
        assert_true(isinstance(hdr.matrix, ci.Cifti2Matrix))
        assert_true(isinstance(hdr.matrix.metadata, ci.Cifti2MetaData))
        for mim in hdr.matrix:
            assert_true(isinstance(mim, ci.Cifti2MatrixIndicesMap))
            counter[ci.Cifti2MatrixIndicesMap] += 1
            for map_ in mim:
                print(map_)
                if isinstance(map_, ci.Cifti2BrainModel):
                    counter[ci.Cifti2BrainModel] += 1
                    if isinstance(map_.vertex_indices, ci.Cifti2VertexIndices):
                        counter[ci.Cifti2VertexIndices] += 1
                    if isinstance(map_.voxel_indices_ijk, ci.Cifti2VoxelIndicesIJK):
                        counter[ci.Cifti2VoxelIndicesIJK] += 1
                elif isinstance(map_, ci.Cifti2NamedMap):
                    counter[ci.Cifti2NamedMap] += 1
                    assert_true(isinstance(map_.metadata, ci.Cifti2MetaData))
                    if isinstance(map_.label_table, ci.Cifti2LabelTable):
                        counter[ci.Cifti2LabelTable] += 1
                        for label in map_.label_table:
                            assert_true(isinstance(map_.label_table[label], ci.Cifti2Label))
                            counter[ci.Cifti2Label] += 1
                elif isinstance(map_, ci.Cifti2Parcel):
                    counter[ci.Cifti2Parcel] += 1
                    if isinstance(map_.voxel_indices_ijk,
                                  ci.Cifti2VoxelIndicesIJK):
                        counter[ci.Cifti2VoxelIndicesIJK] += 1
                    assert_true(isinstance(map_.vertices, list))
                    for vtcs in map_.vertices:
                        assert_true(isinstance(vtcs, ci.Cifti2Vertices))
                        counter[ci.Cifti2Vertices] += 1
                elif isinstance(map_, ci.Cifti2Surface):
                    counter[ci.Cifti2Surface] += 1
                elif isinstance(map_, ci.Cifti2Volume):
                    counter[ci.Cifti2Volume] += 1
                    if isinstance(map_.transformation_matrix_voxel_indices_ijk_to_xyz,
                                  ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ):
                        counter[ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ] += 1

            assert_equal(list(mim.named_maps), [m_ for m_ in mim if isinstance(m_, ci.Cifti2NamedMap)])
            assert_equal(list(mim.surfaces), [m_ for m_ in mim if isinstance(m_, ci.Cifti2Surface)])
            assert_equal(list(mim.parcels), [m_ for m_ in mim if isinstance(m_, ci.Cifti2Parcel)])
            assert_equal(list(mim.brain_models), [m_ for m_ in mim if isinstance(m_, ci.Cifti2BrainModel)])
            assert_equal([mim.volume] if mim.volume else [], [m_ for m_ in mim if isinstance(m_, ci.Cifti2Volume)])

    for klass, count in counter.items():
        assert_true(count > 0, "No exercise of " + klass.__name__)
