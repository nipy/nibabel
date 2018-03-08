"""Tests the generation of new CIFTI2 files from scratch

Contains a series of functions to create and check each of the 5 CIFTI index
types (i.e. BRAIN_MODELS, PARCELS, SCALARS, LABELS, and SERIES).

These functions are used in the tests to generate most CIFTI file types from
scratch.
"""
import numpy as np

import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory

from nose.tools import assert_true, assert_equal

affine = [[-1.5, 0, 0, 90],
          [0, 1.5, 0, -85],
          [0, 0, 1.5, -71]]

dimensions = (120, 83, 78)

number_of_vertices = 30000

brain_models = [('CIFTI_STRUCTURE_THALAMUS_LEFT', [[60, 60, 60],
                                                   [61, 59, 60],
                                                   [61, 60, 59],
                                                   [80, 90, 92]]),
                ('CIFTI_STRUCTURE_CORTEX_LEFT', [0, 1000, 1301, 19972, 27312])]


def create_geometry_map(applies_to_matrix_dimension):
    voxels = ci.Cifti2VoxelIndicesIJK(brain_models[0][1])
    left_thalamus = ci.Cifti2BrainModel(index_offset=0, index_count=4,
                                        model_type='CIFTI_MODEL_TYPE_VOXELS',
                                        brain_structure=brain_models[0][0],
                                        voxel_indices_ijk=voxels)
    vertices = ci.Cifti2VertexIndices(np.array(brain_models[1][1]))
    left_cortex = ci.Cifti2BrainModel(index_offset=4, index_count=5,
                                      model_type='CIFTI_MODEL_TYPE_SURFACE',
                                      brain_structure=brain_models[1][0],
                                      vertex_indices=vertices)
    left_cortex.surface_number_of_vertices = number_of_vertices
    volume = ci.Cifti2Volume(dimensions,
                     ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3,
                                                                       affine))
    return ci.Cifti2MatrixIndicesMap(applies_to_matrix_dimension,
                                     'CIFTI_INDEX_TYPE_BRAIN_MODELS',
                                     maps=[left_thalamus, left_cortex, volume])


def check_geometry_map(mapping):
    assert_equal(mapping.indices_map_to_data_type,
                 'CIFTI_INDEX_TYPE_BRAIN_MODELS')
    assert_equal(len(list(mapping.brain_models)), 2)
    left_thalamus, left_cortex = mapping.brain_models

    assert_equal(left_thalamus.index_offset, 0)
    assert_equal(left_thalamus.index_count, 4)
    assert_equal(left_thalamus.model_type, 'CIFTI_MODEL_TYPE_VOXELS')
    assert_equal(left_thalamus.brain_structure, brain_models[0][0])
    assert_equal(left_thalamus.vertex_indices, None)
    assert_equal(left_thalamus.surface_number_of_vertices, None)
    assert_equal(left_thalamus.voxel_indices_ijk._indices, brain_models[0][1])

    assert_equal(left_cortex.index_offset, 4)
    assert_equal(left_cortex.index_count, 5)
    assert_equal(left_cortex.model_type, 'CIFTI_MODEL_TYPE_SURFACE')
    assert_equal(left_cortex.brain_structure, brain_models[1][0])
    assert_equal(left_cortex.voxel_indices_ijk, None)
    assert_equal(left_cortex.vertex_indices._indices, brain_models[1][1])
    assert_equal(left_cortex.surface_number_of_vertices, number_of_vertices)

    assert_equal(mapping.volume.volume_dimensions, dimensions)
    assert_true((mapping.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix == affine).all())

parcels = [('volume_parcel', ([[60, 60, 60],
                               [61, 59, 60],
                               [61, 60, 59],
                               [80, 90, 92]], )),
           ('surface_parcel', (('CIFTI_STRUCTURE_CORTEX_LEFT',
                                [0, 1000, 1301, 19972, 27312]),
                               ('CIFTI_STRUCTURE_CORTEX_RIGHT',
                                [0, 100, 381]))),
           ('mixed_parcel', ([[71, 81, 39],
                              [53, 21, 91]],
                             ('CIFTI_STRUCTURE_CORTEX_LEFT', [71, 88, 999])))]


def create_parcel_map(applies_to_matrix_dimension):
    mapping = ci.Cifti2MatrixIndicesMap(applies_to_matrix_dimension,
                                        'CIFTI_INDEX_TYPE_PARCELS')
    for name, elements in parcels:
        surfaces = []
        volume = None
        for element in elements:
            if isinstance(element[0], str):
                surfaces.append(ci.Cifti2Vertices(element[0], element[1]))
            else:
                volume = ci.Cifti2VoxelIndicesIJK(element)
        mapping.append(ci.Cifti2Parcel(name, volume, surfaces))

    mapping.extend([ci.Cifti2Surface('CIFTI_STRUCTURE_CORTEX_%s' % orientation,
                    number_of_vertices) for orientation in ['LEFT', 'RIGHT']])
    mapping.volume = ci.Cifti2Volume(dimensions,
                 ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, affine))
    return mapping


def check_parcel_map(mapping):
    assert_equal(mapping.indices_map_to_data_type, 'CIFTI_INDEX_TYPE_PARCELS')
    assert_equal(len(list(mapping.parcels)), 3)
    for (name, elements), parcel in zip(parcels, mapping.parcels):
        assert_equal(parcel.name, name)
        idx_surface = 0
        for element in elements:
            if isinstance(element[0], str):
                surface = parcel.vertices[idx_surface]
                assert_equal(surface.brain_structure, element[0])
                assert_equal(surface._vertices, element[1])
                idx_surface += 1
            else:
                assert_equal(parcel.voxel_indices_ijk._indices, element)

    for surface, orientation in zip(mapping.surfaces, ('LEFT', 'RIGHT')):
        assert_equal(surface.brain_structure,
                     'CIFTI_STRUCTURE_CORTEX_%s' % orientation)
        assert_equal(surface.surface_number_of_vertices, number_of_vertices)

    assert_equal(mapping.volume.volume_dimensions, dimensions)
    assert_true((mapping.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix == affine).all())


scalars = [('first_name', {'meta_key': 'some_metadata'}),
           ('another name', {})]


def create_scalar_map(applies_to_matrix_dimension):
    maps = [ci.Cifti2NamedMap(name, ci.Cifti2MetaData(meta))
            for name, meta in scalars]
    return ci.Cifti2MatrixIndicesMap(applies_to_matrix_dimension,
                                     'CIFTI_INDEX_TYPE_SCALARS',
                                     maps=maps)


def check_scalar_map(mapping):
    assert_equal(mapping.indices_map_to_data_type, 'CIFTI_INDEX_TYPE_SCALARS')
    assert_equal(len(list(mapping.named_maps)), 2)

    for expected, named_map in zip(scalars, mapping.named_maps):
        assert_equal(named_map.map_name, expected[0])
        if len(expected[1]) == 0:
            assert_equal(named_map.metadata, None)
        else:
            assert_equal(named_map.metadata, expected[1])


labels = [('first_name', {'meta_key': 'some_metadata'},
           {0: ('label0', (0.1, 0.3, 0.2, 0.5)),
            1: ('new_label', (0.5, 0.3, 0.1, 0.4))}),
          ('another name', {}, {0: ('???', (0, 0, 0, 0)),
                                1: ('great region', (0.4, 0.1, 0.23, 0.15))})]


def create_label_map(applies_to_matrix_dimension):
    maps = []
    for name, meta, label in labels:
        label_table = ci.Cifti2LabelTable()
        for key, (tag, rgba) in label.items():
            label_table[key] = ci.Cifti2Label(key, tag, *rgba)
        maps.append(ci.Cifti2NamedMap(name, ci.Cifti2MetaData(meta),
                                      label_table))
    return ci.Cifti2MatrixIndicesMap(applies_to_matrix_dimension,
                                     'CIFTI_INDEX_TYPE_LABELS',
                                     maps=maps)


def check_label_map(mapping):
    assert_equal(mapping.indices_map_to_data_type, 'CIFTI_INDEX_TYPE_LABELS')
    assert_equal(len(list(mapping.named_maps)), 2)

    for expected, named_map in zip(scalars, mapping.named_maps):
        assert_equal(named_map.map_name, expected[0])
        if len(expected[1]) == 0:
            assert_equal(named_map.metadata, None)
        else:
            assert_equal(named_map.metadata, expected[1])


def create_series_map(applies_to_matrix_dimension):
    return ci.Cifti2MatrixIndicesMap(applies_to_matrix_dimension,
                                     'CIFTI_INDEX_TYPE_SERIES',
                                     number_of_series_points=13,
                                     series_exponent=-3, series_start=18.2,
                                     series_step=10.5, series_unit='SECOND')


def check_series_map(mapping):
    assert_equal(mapping.indices_map_to_data_type, 'CIFTI_INDEX_TYPE_SERIES')
    assert_equal(mapping.number_of_series_points, 13)
    assert_equal(mapping.series_exponent, -3)
    assert_equal(mapping.series_start, 18.2)
    assert_equal(mapping.series_step, 10.5)
    assert_equal(mapping.series_unit, 'SECOND')


def test_dtseries():
    series_map = create_series_map((0, ))
    geometry_map = create_geometry_map((1, ))
    matrix = ci.Cifti2Matrix()
    matrix.append(series_map)
    matrix.append(geometry_map)
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(13, 9)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES')

    with InTemporaryDirectory():
        ci.save(img, 'test.dtseries.nii')
        img2 = nib.load('test.dtseries.nii')
        assert_equal(img2.nifti_header.get_intent()[0],
                     'ConnDenseSeries')
        assert_true(isinstance(img2, ci.Cifti2Image))
        assert_true((img2.get_data() == data).all())
        check_series_map(img2.header.matrix.get_index_map(0))
        check_geometry_map(img2.header.matrix.get_index_map(1))
        del img2


def test_dscalar():
    scalar_map = create_scalar_map((0, ))
    geometry_map = create_geometry_map((1, ))
    matrix = ci.Cifti2Matrix()
    matrix.append(scalar_map)
    matrix.append(geometry_map)
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(2, 9)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_SCALARS')

    with InTemporaryDirectory():
        ci.save(img, 'test.dscalar.nii')
        img2 = nib.load('test.dscalar.nii')
        assert_equal(img2.nifti_header.get_intent()[0], 'ConnDenseScalar')
        assert_true(isinstance(img2, ci.Cifti2Image))
        assert_true((img2.get_data() == data).all())
        check_scalar_map(img2.header.matrix.get_index_map(0))
        check_geometry_map(img2.header.matrix.get_index_map(1))
        del img2


def test_dlabel():
    label_map = create_label_map((0, ))
    geometry_map = create_geometry_map((1, ))
    matrix = ci.Cifti2Matrix()
    matrix.append(label_map)
    matrix.append(geometry_map)
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(2, 9)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_LABELS')

    with InTemporaryDirectory():
        ci.save(img, 'test.dlabel.nii')
        img2 = nib.load('test.dlabel.nii')
        assert_equal(img2.nifti_header.get_intent()[0], 'ConnDenseLabel')
        assert_true(isinstance(img2, ci.Cifti2Image))
        assert_true((img2.get_data() == data).all())
        check_label_map(img2.header.matrix.get_index_map(0))
        check_geometry_map(img2.header.matrix.get_index_map(1))
        del img2


def test_dconn():
    mapping = create_geometry_map((0, 1))
    matrix = ci.Cifti2Matrix()
    matrix.append(mapping)
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(9, 9)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE')

    with InTemporaryDirectory():
        ci.save(img, 'test.dconn.nii')
        img2 = nib.load('test.dconn.nii')
        assert_equal(img2.nifti_header.get_intent()[0], 'ConnDense')
        assert_true(isinstance(img2, ci.Cifti2Image))
        assert_true((img2.get_data() == data).all())
        assert_equal(img2.header.matrix.get_index_map(0),
                     img2.header.matrix.get_index_map(1))
        check_geometry_map(img2.header.matrix.get_index_map(0))
        del img2


def test_ptseries():
    series_map = create_series_map((0, ))
    parcel_map = create_parcel_map((1, ))
    matrix = ci.Cifti2Matrix()
    matrix.append(series_map)
    matrix.append(parcel_map)
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(13, 3)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_PARCELLATED_SERIES')

    with InTemporaryDirectory():
        ci.save(img, 'test.ptseries.nii')
        img2 = nib.load('test.ptseries.nii')
        assert_equal(img2.nifti_header.get_intent()[0], 'ConnParcelSries')
        assert_true(isinstance(img2, ci.Cifti2Image))
        assert_true((img2.get_data() == data).all())
        check_series_map(img2.header.matrix.get_index_map(0))
        check_parcel_map(img2.header.matrix.get_index_map(1))
        del img2


def test_pscalar():
    scalar_map = create_scalar_map((0, ))
    parcel_map = create_parcel_map((1, ))
    matrix = ci.Cifti2Matrix()
    matrix.append(scalar_map)
    matrix.append(parcel_map)
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(2, 3)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_PARCELLATED_SCALAR')

    with InTemporaryDirectory():
        ci.save(img, 'test.pscalar.nii')
        img2 = nib.load('test.pscalar.nii')
        assert_equal(img2.nifti_header.get_intent()[0], 'ConnParcelScalr')
        assert_true(isinstance(img2, ci.Cifti2Image))
        assert_true((img2.get_data() == data).all())
        check_scalar_map(img2.header.matrix.get_index_map(0))
        check_parcel_map(img2.header.matrix.get_index_map(1))
        del img2


def test_pdconn():
    geometry_map = create_geometry_map((0, ))
    parcel_map = create_parcel_map((1, ))
    matrix = ci.Cifti2Matrix()
    matrix.append(geometry_map)
    matrix.append(parcel_map)
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(2, 3)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_PARCELLATED_DENSE')

    with InTemporaryDirectory():
        ci.save(img, 'test.pdconn.nii')
        img2 = ci.load('test.pdconn.nii')
        assert_equal(img2.nifti_header.get_intent()[0], 'ConnParcelDense')
        assert_true(isinstance(img2, ci.Cifti2Image))
        assert_true((img2.get_data() == data).all())
        check_geometry_map(img2.header.matrix.get_index_map(0))
        check_parcel_map(img2.header.matrix.get_index_map(1))
        del img2


def test_dpconn():
    parcel_map = create_parcel_map((0, ))
    geometry_map = create_geometry_map((1, ))
    matrix = ci.Cifti2Matrix()
    matrix.append(parcel_map)
    matrix.append(geometry_map)
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(2, 3)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_PARCELLATED')

    with InTemporaryDirectory():
        ci.save(img, 'test.dpconn.nii')
        img2 = ci.load('test.dpconn.nii')
        assert_equal(img2.nifti_header.get_intent()[0], 'ConnDenseParcel')
        assert_true(isinstance(img2, ci.Cifti2Image))
        assert_true((img2.get_data() == data).all())
        check_parcel_map(img2.header.matrix.get_index_map(0))
        check_geometry_map(img2.header.matrix.get_index_map(1))
        del img2


def test_plabel():
    label_map = create_label_map((0, ))
    parcel_map = create_parcel_map((1, ))
    matrix = ci.Cifti2Matrix()
    matrix.append(label_map)
    matrix.append(parcel_map)
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(2, 3)
    img = ci.Cifti2Image(data, hdr)

    with InTemporaryDirectory():
        ci.save(img, 'test.plabel.nii')
        img2 = ci.load('test.plabel.nii')
        assert_equal(img.nifti_header.get_intent()[0], 'ConnUnknown')
        assert_true(isinstance(img2, ci.Cifti2Image))
        assert_true((img2.get_data() == data).all())
        check_label_map(img2.header.matrix.get_index_map(0))
        check_parcel_map(img2.header.matrix.get_index_map(1))
        del img2


def test_pconn():
    mapping = create_parcel_map((0, 1))
    matrix = ci.Cifti2Matrix()
    matrix.append(mapping)
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(3, 3)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_PARCELLATED')

    with InTemporaryDirectory():
        ci.save(img, 'test.pconn.nii')
        img2 = ci.load('test.pconn.nii')
        assert_equal(img.nifti_header.get_intent()[0], 'ConnParcels')
        assert_true(isinstance(img2, ci.Cifti2Image))
        assert_true((img2.get_data() == data).all())
        assert_equal(img2.header.matrix.get_index_map(0),
                     img2.header.matrix.get_index_map(1))
        check_parcel_map(img2.header.matrix.get_index_map(0))
        del img2


def test_pconnseries():
    parcel_map = create_parcel_map((0, 1))
    series_map = create_series_map((2, ))

    matrix = ci.Cifti2Matrix()
    matrix.append(parcel_map)
    matrix.append(series_map)
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(3, 3, 13)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_PARCELLATED_'
                                'PARCELLATED_SERIES')

    with InTemporaryDirectory():
        ci.save(img, 'test.pconnseries.nii')
        img2 = ci.load('test.pconnseries.nii')
        assert_equal(img.nifti_header.get_intent()[0], 'ConnPPSr')
        assert_true(isinstance(img2, ci.Cifti2Image))
        assert_true((img2.get_data() == data).all())
        assert_equal(img2.header.matrix.get_index_map(0),
                     img2.header.matrix.get_index_map(1))
        check_parcel_map(img2.header.matrix.get_index_map(0))
        check_series_map(img2.header.matrix.get_index_map(2))
        del img2


def test_pconnscalar():
    parcel_map = create_parcel_map((0, 1))
    scalar_map = create_scalar_map((2, ))

    matrix = ci.Cifti2Matrix()
    matrix.append(parcel_map)
    matrix.append(scalar_map)
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(3, 3, 13)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_PARCELLATED_'
                                'PARCELLATED_SCALAR')

    with InTemporaryDirectory():
        ci.save(img, 'test.pconnscalar.nii')
        img2 = ci.load('test.pconnscalar.nii')
        assert_equal(img.nifti_header.get_intent()[0], 'ConnPPSc')
        assert_true(isinstance(img2, ci.Cifti2Image))
        assert_true((img2.get_data() == data).all())
        assert_equal(img2.header.matrix.get_index_map(0),
                     img2.header.matrix.get_index_map(1))

        check_parcel_map(img2.header.matrix.get_index_map(0))
        check_scalar_map(img2.header.matrix.get_index_map(2))
        del img2
