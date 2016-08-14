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
import io

from distutils.version import LooseVersion

import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.cifti2.parse_cifti2 import _Cifti2AsNiftiHeader

from nibabel.tmpdirs import InTemporaryDirectory
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from nibabel.tests.test_nifti2 import TestNifti2SingleHeader

from numpy.testing import assert_array_almost_equal
from nose.tools import (assert_true, assert_equal, assert_raises)

NIBABEL_TEST_DATA = pjoin(dirname(nib.__file__), 'tests', 'data')
NIFTI2_DATA = pjoin(NIBABEL_TEST_DATA, 'example_nifti2.nii.gz')

CIFTI2_DATA = pjoin(get_nibabel_data(), 'nitest-cifti2')

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


def test_read_nifti2():
    # Error trying to read a CIFTI2 image from a NIfTI2-only image.
    filemap = ci.Cifti2Image.make_file_map()
    for k in filemap:
        filemap[k].fileobj = io.open(NIFTI2_DATA)
    assert_raises(ValueError, ci.Cifti2Image.from_file_map, filemap)


@needs_nibabel_data('nitest-cifti2')
def test_read_internal():
    img2 = ci.load(DATA_FILE6)
    assert_true(isinstance(img2.header, ci.Cifti2Header))
    assert_equal(img2.shape, (1, 91282))


@needs_nibabel_data('nitest-cifti2')
def test_read_and_proxies():
    img2 = nib.load(DATA_FILE6)
    assert_true(isinstance(img2.header, ci.Cifti2Header))
    assert_equal(img2.shape, (1, 91282))
    # While we cannot reshape arrayproxies, all images are in-memory
    assert_true(img2.in_memory)
    data = img2.get_data()
    assert_true(data is img2.dataobj)
    # Uncaching has no effect, images are always array images
    img2.uncache()
    assert_true(data is img2.get_data())


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
            ci.save(img, 'test.nii')
            img2 = ci.load('test.nii')
            assert_equal(len(img.header.matrix),
                         len(img2.header.matrix))
            # Order should be preserved in load/save
            for mim1, mim2 in zip(img.header.matrix,
                                  img2.header.matrix):
                named_maps1 = [m_ for m_ in mim1
                               if isinstance(m_, ci.Cifti2NamedMap)]
                named_maps2 = [m_ for m_ in mim2
                               if isinstance(m_, ci.Cifti2NamedMap)]
                assert_equal(len(named_maps1), len(named_maps2))
                for map1, map2 in zip(named_maps1, named_maps2):
                    assert_equal(map1.map_name, map2.map_name)
                    if map1.label_table is None:
                        assert_true(map2.label_table is None)
                    else:
                        assert_equal(len(map1.label_table),
                                     len(map2.label_table))
            assert_array_almost_equal(img.dataobj, img2.dataobj)


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
                named_maps1 = [m_ for m_ in mim1
                               if isinstance(m_, ci.Cifti2NamedMap)]
                named_maps2 = [m_ for m_ in mim2
                               if isinstance(m_, ci.Cifti2NamedMap)]
                assert_equal(len(named_maps1), len(named_maps2))
                for map1, map2 in zip(named_maps1, named_maps2):
                    assert_equal(map1.map_name, map2.map_name)
                    if map1.label_table is None:
                        assert_true(map2.label_table is None)
                    else:
                        assert_equal(len(map1.label_table),
                                     len(map2.label_table))
            assert_array_almost_equal(img.dataobj, img2.dataobj)


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
                    if isinstance(map_.voxel_indices_ijk,
                                  ci.Cifti2VoxelIndicesIJK):
                        counter[ci.Cifti2VoxelIndicesIJK] += 1
                elif isinstance(map_, ci.Cifti2NamedMap):
                    counter[ci.Cifti2NamedMap] += 1
                    assert_true(isinstance(map_.metadata, ci.Cifti2MetaData))
                    if isinstance(map_.label_table, ci.Cifti2LabelTable):
                        counter[ci.Cifti2LabelTable] += 1
                        for label in map_.label_table:
                            assert_true(isinstance(map_.label_table[label],
                                                   ci.Cifti2Label))
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

            assert_equal(list(mim.named_maps),
                         [m_ for m_ in mim if isinstance(m_, ci.Cifti2NamedMap)])
            assert_equal(list(mim.surfaces),
                         [m_ for m_ in mim if isinstance(m_, ci.Cifti2Surface)])
            assert_equal(list(mim.parcels),
                         [m_ for m_ in mim if isinstance(m_, ci.Cifti2Parcel)])
            assert_equal(list(mim.brain_models),
                         [m_ for m_ in mim if isinstance(m_, ci.Cifti2BrainModel)])
            assert_equal([mim.volume] if mim.volume else [],
                         [m_ for m_ in mim if isinstance(m_, ci.Cifti2Volume)])

    for klass, count in counter.items():
        assert_true(count > 0, "No exercise of " + klass.__name__)


class TestCifti2SingleHeader(TestNifti2SingleHeader):
    header_class = _Cifti2AsNiftiHeader
    _pixdim_message = 'pixdim[1,2,3] should be zero or positive'

    def test_pixdim_checks(self):
        hdr_t = self.header_class()
        for i in (1, 2, 3):
            hdr = hdr_t.copy()
            hdr['pixdim'][i] = -1
            assert_equal(self._dxer(hdr), self._pixdim_message)

    def test_nifti_qfac_checks(self):
        # Test qfac is 1 or -1 or 0
        hdr = self.header_class()
        # 1, 0, -1 all OK
        hdr['pixdim'][0] = 1
        self.log_chk(hdr, 0)
        hdr['pixdim'][0] = 0
        self.log_chk(hdr, 0)
        hdr['pixdim'][0] = -1
        self.log_chk(hdr, 0)
        # Anything else is not
        hdr['pixdim'][0] = 2
        fhdr, message, raiser = self.log_chk(hdr, 20)
        assert_equal(fhdr['pixdim'][0], 1)
        assert_equal(message,
                     'pixdim[0] (qfac) should be 1 '
                     '(default) or 0 or -1; setting qfac to 1')

    def test_pixdim_log_checks(self):
        # pixdim can be zero or positive
        HC = self.header_class
        hdr = HC()
        hdr['pixdim'][1] = -2  # severity 35
        fhdr, message, raiser = self.log_chk(hdr, 35)
        assert_equal(fhdr['pixdim'][1], 2)
        assert_equal(message, self._pixdim_message +
                     '; setting to abs of pixdim values')
        assert_raises(*raiser)
        hdr = HC()
        hdr['pixdim'][1:4] = 0  # No error or warning
        fhdr, message, raiser = self.log_chk(hdr, 0)
        assert_equal(raiser, ())
