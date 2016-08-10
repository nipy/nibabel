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

from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_raises)

IO_DATA_PATH = pjoin(dirname(__file__), 'data')
DATA_FILE1 = pjoin(IO_DATA_PATH, '')
DATA_FILE2 = pjoin(IO_DATA_PATH,
                   'Conte69.MyelinAndCorrThickness.32k_fs_LR.dscalar.nii')
DATA_FILE3 = pjoin(IO_DATA_PATH,
                   'Conte69.MyelinAndCorrThickness.32k_fs_LR.dtseries.nii')
DATA_FILE4 = pjoin(IO_DATA_PATH,
                   'Conte69.MyelinAndCorrThickness.32k_fs_LR.ptseries.nii')
DATA_FILE5 = pjoin(IO_DATA_PATH,
                   'Conte69.parcellations_VGD11b.32k_fs_LR.dlabel.nii')
DATA_FILE6 = pjoin(IO_DATA_PATH, 'ones.dscalar.nii')
datafiles = [DATA_FILE2, DATA_FILE3, DATA_FILE4, DATA_FILE5, DATA_FILE6]


def test_read_ordering():
    img2 = nib.load(DATA_FILE6)
    assert_equal(img2.data.shape, (1, 91282))


def test_version():
    for i, dat in enumerate(datafiles):
        img = nib.load(dat)
        assert_equal(LooseVersion(img.header.version), LooseVersion('2'))

'''
def test_dataarray1():
    img1 = gi.read(DATA_FILE1)
    # Round trip
    with InTemporaryDirectory():
        gi.write(img1, 'test.gii')
        bimg = gi.read('test.gii')
    for img in (img1, bimg):
        assert_array_almost_equal(img.darrays[0].data, DATA_FILE1_darr1)
        assert_array_almost_equal(img.darrays[1].data, DATA_FILE1_darr2)
        me=img.darrays[0].meta.get_metadata()
        assert_true('AnatomicalStructurePrimary' in me)
        assert_true('AnatomicalStructureSecondary' in me)
        assert_equal(me['AnatomicalStructurePrimary'], 'CortexLeft')
        assert_array_almost_equal(img.darrays[0].coordsys.xform, np.eye(4,4))
        assert_equal(xform_codes.niistring[img.darrays[0].coordsys.dataspace],'NIFTI_XFORM_TALAIRACH')
        assert_equal(xform_codes.niistring[img.darrays[0].coordsys.xformspace],'NIFTI_XFORM_TALAIRACH')
'''


def test_readwritedata():
    with InTemporaryDirectory():
        for name in datafiles:
            img = nib.load(name)
            nib.save(img, 'test.nii')
            img2 = nib.load('test.nii')
            assert_equal(len(img.header.matrix.mims),
                         len(img2.header.matrix.mims))
            # Order should be preserved in load/save
            for mim1, mim2 in zip(img.header.matrix.mims,
                                  img2.header.matrix.mims):
                assert_equal(len(mim1.named_maps), len(mim2.named_maps))
                for map1, map2 in zip(mim1.named_maps, mim2.named_maps):
                    assert_equal(map1.map_name, map2.map_name)
                    if map1.label_table is None:
                        assert_true(map2.label_table is None)
                    else:
                        assert_equal(len(map1.label_table.labels),
                                     len(map2.label_table.labels))
            assert_array_almost_equal(img.data, img2.data)


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
        hdr = nib.load(name).header
        # Matrix and MetaData aren't conditional, so don't bother counting
        assert_true(isinstance(hdr.matrix, ci.Cifti2Matrix))
        assert_true(isinstance(hdr.matrix.metadata, ci.Cifti2MetaData))
        assert_true(isinstance(hdr.matrix.mims, list))
        for mim in hdr.matrix.mims:
            assert_true(isinstance(mim, ci.Cifti2MatrixIndicesMap))
            counter[ci.Cifti2MatrixIndicesMap] += 1
            assert_true(isinstance(mim.brain_models, list))
            for bm in mim.brain_models:
                assert_true(isinstance(bm, ci.Cifti2BrainModel))
                counter[ci.Cifti2BrainModel] += 1
                if isinstance(bm.vertex_indices, ci.Cifti2VertexIndices):
                    counter[ci.Cifti2VertexIndices] += 1
                if isinstance(bm.voxel_indices_ijk, ci.Cifti2VoxelIndicesIJK):
                    counter[ci.Cifti2VoxelIndicesIJK] += 1
            assert_true(isinstance(mim.named_maps, list))
            for nm in mim.named_maps:
                assert_true(isinstance(nm, ci.Cifti2NamedMap))
                counter[ci.Cifti2NamedMap] += 1
                assert_true(isinstance(nm.metadata, ci.Cifti2MetaData))
                if isinstance(nm.label_table, ci.Cifti2LabelTable):
                    counter[ci.Cifti2LabelTable] += 1
                    assert_true(isinstance(nm.label_table.labels, list))
                    for label in nm.label_table.labels:
                        assert_true(isinstance(label, ci.Cifti2Label))
                        counter[ci.Cifti2Label] += 1
            assert_true(isinstance(mim.parcels, list))
            for parc in mim.parcels:
                assert_true(isinstance(parc, ci.Cifti2Parcel))
                counter[ci.Cifti2Parcel] += 1
                if isinstance(parc.voxel_indices_ijk,
                              ci.Cifti2VoxelIndicesIJK):
                    counter[ci.Cifti2VoxelIndicesIJK] += 1
                assert_true(isinstance(parc.vertices, list))
                for vtcs in parc.vertices:
                    assert_true(isinstance(vtcs, ci.Cifti2Vertices))
                    counter[ci.Cifti2Vertices] += 1
            assert_true(isinstance(mim.surfaces, list))
            for surf in mim.surfaces:
                assert_true(isinstance(surf, ci.Cifti2Surface))
                counter[ci.Cifti2Surface] += 1
            if isinstance(mim.volume, ci.Cifti2Volume):
                counter[ci.Cifti2Volume] += 1
                if isinstance(mim.volume.transformation_matrix_voxel_indices_ijk_to_xyz,
                              ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ):
                    counter[ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ] += 1
    for klass, count in counter.items():
        assert_true(count > 0, "No exercise of " + klass.__name__)
