""" Testing CIFTI2 objects
"""
import collections
from xml.etree import ElementTree

import numpy as np

from nibabel import cifti2 as ci
from nibabel.nifti2 import Nifti2Header
from nibabel.cifti2.cifti2 import _float_01, _value_if_klass, Cifti2HeaderError

from nose.tools import assert_true, assert_equal, assert_raises, assert_is_none

from nibabel.tests.test_dataobj_images import TestDataobjAPI as _TDA


def compare_xml_leaf(str1, str2):
    x1 = ElementTree.fromstring(str1)
    x2 = ElementTree.fromstring(str2)
    if len(x1) > 0 or len(x2) > 0:
        raise ValueError

    test = (x1.tag == x2.tag) and (x1.attrib == x2.attrib) and (x1.text == x2.text)
    print((x1.tag, x1.attrib, x1.text))
    print((x2.tag, x2.attrib, x2.text))
    return test


def test_value_if_klass():
    assert_equal(_value_if_klass(None, list), None)
    assert_equal(_value_if_klass([1], list), [1])
    assert_raises(ValueError, _value_if_klass, 1, list)


def test_cifti2_metadata():
    md = ci.Cifti2MetaData(metadata={'a': 'aval'})
    assert_equal(len(md), 1)
    assert_equal(list(iter(md)), ['a'])
    assert_equal(md['a'], 'aval')
    assert_equal(md.data, dict([('a', 'aval')]))

    md = ci.Cifti2MetaData()
    assert_equal(len(md), 0)
    assert_equal(list(iter(md)), [])
    assert_equal(md.data, {})
    assert_raises(ValueError, md.difference_update, None)

    md['a'] = 'aval'
    assert_equal(md['a'], 'aval')
    assert_equal(len(md), 1)
    assert_equal(md.data, dict([('a', 'aval')]))

    del md['a']
    assert_equal(len(md), 0)

    metadata_test = [('a', 'aval'), ('b', 'bval')]
    md.update(metadata_test)
    assert_equal(md.data, dict(metadata_test))

    assert_equal(list(iter(md)), list(iter(collections.OrderedDict(metadata_test))))
    
    md.update({'a': 'aval', 'b': 'bval'})
    assert_equal(md.data, dict(metadata_test))

    md.update({'a': 'aval', 'd': 'dval'})
    assert_equal(md.data, dict(metadata_test + [('d', 'dval')]))

    md.difference_update({'a': 'aval', 'd': 'dval'})
    assert_equal(md.data, dict(metadata_test[1:]))

    assert_raises(KeyError, md.difference_update, {'a': 'aval', 'd': 'dval'})
    assert_equal(md.to_xml().decode('utf-8'),
                 '<MetaData><MD><Name>b</Name><Value>bval</Value></MD></MetaData>')


def test__float_01():
    assert_equal(_float_01(0), 0)
    assert_equal(_float_01(1), 1)
    assert_equal(_float_01('0'), 0)
    assert_equal(_float_01('0.2'), 0.2)
    assert_raises(ValueError, _float_01, 1.1)
    assert_raises(ValueError, _float_01, -0.1)
    assert_raises(ValueError, _float_01, 2)
    assert_raises(ValueError, _float_01, -1)
    assert_raises(ValueError, _float_01, 'foo')


def test_cifti2_labeltable():
    lt = ci.Cifti2LabelTable()
    assert_equal(len(lt), 0)
    assert_raises(ci.Cifti2HeaderError, lt.to_xml)
    assert_raises(ci.Cifti2HeaderError, lt._to_xml_element)

    label = ci.Cifti2Label(label='Test', key=0)
    lt[0] = label
    assert_equal(len(lt), 1)
    assert_equal(dict(lt), {label.key: label})

    lt.clear()
    lt.append(label)
    assert_equal(len(lt), 1)
    assert_equal(dict(lt), {label.key: label})

    lt.clear()
    test_tuple = (label.label, label.red, label.green, label.blue, label.alpha)
    lt[label.key] = test_tuple
    assert_equal(len(lt), 1)
    v = lt[label.key]
    assert_equal(
        (v.label, v.red, v.green, v.blue, v.alpha),
        test_tuple
    )

    assert_raises(ValueError, lt.__setitem__, 1, label)
    assert_raises(ValueError, lt.__setitem__, 0, test_tuple[:-1])
    assert_raises(ValueError, lt.__setitem__, 0, ('foo', 1.1, 0, 0, 1))
    assert_raises(ValueError, lt.__setitem__, 0, ('foo', 1.0, -1, 0, 1))
    assert_raises(ValueError, lt.__setitem__, 0, ('foo', 1.0, 0, -0.1, 1))


def test_cifti2_label():
    lb = ci.Cifti2Label()
    lb.label = 'Test'
    lb.key = 0
    assert_equal(lb.rgba, (0, 0, 0, 0))
    assert_true(compare_xml_leaf(
        lb.to_xml().decode('utf-8'),
        "<Label Key='0' Red='0' Green='0' Blue='0' Alpha='0'>Test</Label>"
    ))

    lb.red = 0
    lb.green = 0.1
    lb.blue = 0.2
    lb.alpha = 0.3
    assert_equal(lb.rgba, (0, 0.1, 0.2, 0.3))

    assert_true(compare_xml_leaf(
        lb.to_xml().decode('utf-8'),
        "<Label Key='0' Red='0' Green='0.1' Blue='0.2' Alpha='0.3'>Test</Label>"
    ))

    lb.red = 10
    assert_raises(ci.Cifti2HeaderError, lb.to_xml)
    lb.red = 0

    lb.key = 'a'
    assert_raises(ci.Cifti2HeaderError, lb.to_xml)
    lb.key = 0


def test_cifti2_parcel():
    pl = ci.Cifti2Parcel()
    assert_raises(ci.Cifti2HeaderError, pl.to_xml)
    assert_raises(TypeError, pl.append_cifti_vertices, None)

    assert_raises(ValueError, ci.Cifti2Parcel, **{'vertices': [1, 2, 3]})
    pl = ci.Cifti2Parcel(name='region',
                         voxel_indices_ijk=ci.Cifti2VoxelIndicesIJK([[1, 2, 3]]),
                         vertices=[ci.Cifti2Vertices([0, 1, 2])])
    pl.pop_cifti2_vertices(0)
    assert_equal(len(pl.vertices), 0)
    assert_equal(
        pl.to_xml().decode('utf-8'),
        '<Parcel Name="region"><VoxelIndicesIJK>1 2 3</VoxelIndicesIJK></Parcel>'
    )


def test_cifti2_vertices():
    vs = ci.Cifti2Vertices()
    assert_raises(ci.Cifti2HeaderError, vs.to_xml)
    vs.brain_structure = 'CIFTI_STRUCTURE_OTHER'
    assert_equal(
        vs.to_xml().decode('utf-8'),
        '<Vertices BrainStructure="CIFTI_STRUCTURE_OTHER" />'
    )
    assert_equal(len(vs), 0)
    vs.extend(np.array([0, 1, 2]))
    assert_equal(len(vs), 3)
    assert_raises(ValueError, vs.__setitem__, 1, 'a')
    assert_raises(ValueError, vs.insert, 1, 'a')
    assert_equal(
        vs.to_xml().decode('utf-8'),
        '<Vertices BrainStructure="CIFTI_STRUCTURE_OTHER">0 1 2</Vertices>'
    )

    vs[0] = 10
    assert_equal(vs[0], 10)
    assert_equal(len(vs), 3)
    vs = ci.Cifti2Vertices(vertices=[0, 1, 2])
    assert_equal(len(vs), 3)


def test_cifti2_transformationmatrixvoxelindicesijktoxyz():
    tr = ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ()
    assert_raises(ci.Cifti2HeaderError, tr.to_xml)


def test_cifti2_surface():
    s = ci.Cifti2Surface()
    assert_raises(ci.Cifti2HeaderError, s.to_xml)


def test_cifti2_volume():
    vo = ci.Cifti2Volume()
    assert_raises(ci.Cifti2HeaderError, vo.to_xml)


def test_cifti2_vertexindices():
    vi = ci.Cifti2VertexIndices()
    assert_equal(len(vi), 0)
    assert_raises(ci.Cifti2HeaderError, vi.to_xml)
    vi.extend(np.array([0, 1, 2]))
    assert_equal(len(vi), 3)
    assert_equal(
        vi.to_xml().decode('utf-8'),
        '<VertexIndices>0 1 2</VertexIndices>'
    )
    assert_raises(ValueError, vi.__setitem__, 0, 'a')
    vi[0] = 10
    assert_equal(vi[0], 10)
    assert_equal(len(vi), 3)


def test_cifti2_voxelindicesijk():
    vi = ci.Cifti2VoxelIndicesIJK()
    assert_raises(ci.Cifti2HeaderError, vi.to_xml)

    vi = ci.Cifti2VoxelIndicesIJK()
    assert_equal(len(vi), 0)
    assert_raises(ci.Cifti2HeaderError, vi.to_xml)
    vi.extend(np.array([[0, 1, 2]]))
    assert_equal(len(vi), 1)
    assert_equal(vi[0], [0, 1, 2])
    vi.append([3, 4, 5])
    assert_equal(len(vi), 2)
    vi.append([6, 7, 8])
    assert_equal(len(vi), 3)
    del vi[-1]
    assert_equal(len(vi), 2)

    assert_equal(vi[1], [3, 4, 5])
    vi[1] = [3, 4, 6]
    assert_equal(vi[1], [3, 4, 6])
    assert_raises(ValueError, vi.__setitem__, 'a', [1, 2, 3])
    assert_raises(TypeError, vi.__setitem__, [1, 2], [1, 2, 3])
    assert_raises(ValueError, vi.__setitem__, 1, [2, 3])
    assert_equal(vi[1, 1], 4)
    assert_raises(ValueError, vi.__setitem__, [1, 1], 'a')
    assert_equal(vi[0, 1:], [1, 2])
    vi[0, 1] = 10
    assert_equal(vi[0, 1], 10)
    vi[0, 1] = 1

    #test for vi[:, 0] and other slices
    assert_raises(NotImplementedError, vi.__getitem__, (slice(None), 0))
    assert_raises(NotImplementedError, vi.__setitem__, (slice(None), 0), 0)
    assert_raises(NotImplementedError, vi.__delitem__, (slice(None), 0))
    assert_raises(ValueError, vi.__getitem__, (0, 0, 0))
    assert_raises(ValueError, vi.__setitem__, (0, 0, 0), 0)

    assert_equal(
        vi.to_xml().decode('utf-8'),
        '<VoxelIndicesIJK>0 1 2\n3 4 6</VoxelIndicesIJK>'
    )
    assert_raises(TypeError, ci.Cifti2VoxelIndicesIJK, [0, 1])
    vi = ci.Cifti2VoxelIndicesIJK([[1, 2, 3]])
    assert_equal(len(vi), 1)


def test_matrixindicesmap():
    mim = ci.Cifti2MatrixIndicesMap(0, 'CIFTI_INDEX_TYPE_LABELS')
    volume = ci.Cifti2Volume()
    volume2 = ci.Cifti2Volume()
    parcel = ci.Cifti2Parcel()

    assert_is_none(mim.volume)
    mim.append(volume)
    mim.append(parcel)


    assert_equal(mim.volume, volume)
    assert_raises(ci.Cifti2HeaderError, mim.insert, 0, volume)
    assert_raises(ci.Cifti2HeaderError, mim.__setitem__, 1, volume)

    mim[0] = volume2
    assert_equal(mim.volume, volume2)

    del mim.volume
    assert_is_none(mim.volume)
    assert_raises(ValueError, delattr, mim, 'volume')

    mim.volume = volume
    assert_equal(mim.volume, volume)
    mim.volume = volume2
    assert_equal(mim.volume, volume2)

    assert_raises(ValueError, setattr, mim, 'volume', parcel)


def test_matrix():
    m = ci.Cifti2Matrix()
    assert_raises(TypeError, m, setattr, 'metadata', ci.Cifti2Parcel())
    assert_raises(TypeError, m.__setitem__, 0, ci.Cifti2Parcel())
    assert_raises(TypeError, m.insert, 0, ci.Cifti2Parcel())

    mim_none = ci.Cifti2MatrixIndicesMap(None, 'CIFTI_INDEX_TYPE_LABELS')
    mim_0 = ci.Cifti2MatrixIndicesMap(0, 'CIFTI_INDEX_TYPE_LABELS')
    mim_1 = ci.Cifti2MatrixIndicesMap(1, 'CIFTI_INDEX_TYPE_LABELS')
    mim_01 = ci.Cifti2MatrixIndicesMap([0, 1], 'CIFTI_INDEX_TYPE_LABELS')

    assert_raises(ci.Cifti2HeaderError, m.insert, 0, mim_none)
    assert_equal(m.mapped_indices, [])
   
    h = ci.Cifti2Header(matrix=m)
    assert_equal(m.mapped_indices, [])
    m.insert(0, mim_0)
    assert_equal(h.mapped_indices, [0])
    assert_equal(h.number_of_mapped_indices, 1)
    assert_raises(ci.Cifti2HeaderError, m.insert, 0, mim_0)
    assert_raises(ci.Cifti2HeaderError, m.insert, 0, mim_01)
    m[0] = mim_1
    assert_equal(list(m.mapped_indices), [1])
    m.insert(0, mim_0)
    assert_equal(list(sorted(m.mapped_indices)), [0, 1])
    assert_equal(h.number_of_mapped_indices, 2)
    assert_equal(h.get_index_map(0), mim_0)
    assert_equal(h.get_index_map(1), mim_1)
    assert_raises(ci.Cifti2HeaderError, h.get_index_map, 2)


def test_underscoring():
    # Pairs taken from inflection tests
    # https://github.com/jpvanhal/inflection/blob/663982e/test_inflection.py#L113-L125
    pairs = (("Product", "product"),
             ("SpecialGuest", "special_guest"),
             ("ApplicationController", "application_controller"),
             ("Area51Controller", "area51_controller"),
             ("HTMLTidy", "html_tidy"),
             ("HTMLTidyGenerator", "html_tidy_generator"),
             ("FreeBSD", "free_bsd"),
             ("HTML", "html"),
            )

    for camel, underscored in pairs:
        assert_equal(ci.cifti2._underscore(camel), underscored)


class TestCifti2ImageAPI(_TDA):
    """ Basic validation for Cifti2Image instances
    """
    # A callable returning an image from ``image_maker(data, header)``
    image_maker = ci.Cifti2Image
    # A callable returning a header from ``header_maker()``
    header_maker = ci.Cifti2Header
    # A callable returning a nifti header
    ni_header_maker = Nifti2Header
    example_shapes = ((2,), (2, 3), (2, 3, 4))
    standard_extension = '.nii'

    def make_imaker(self, arr, header=None, ni_header=None):
        return lambda: self.image_maker(arr.copy(), header, ni_header)
