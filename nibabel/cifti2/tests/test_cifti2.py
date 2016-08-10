""" Testing gifti objects
"""

import numpy as np
from lxml import etree

from ...nifti1 import data_type_codes, intent_codes

from ... import cifti2 as ci

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_equal, assert_raises



def compare_xml_leaf(str1, str2):
    x1 = etree.fromstring(str1)
    x2 = etree.fromstring(str2)
    if len(x1.getchildren()) > 0 or len(x2.getchildren()) > 0:
        raise ValueError

    return (x1.tag == x2.tag) and (x1.attrib and x2.attrib)

def test_cifti2_MetaData():
    md = ci.Cifti2MetaData()
    assert_equal(md.data, {})
    assert_raises(ValueError, md.add_metadata, 0)
    assert_raises(ValueError, md.add_metadata, ['a'])
    assert_raises(ValueError, md.remove_metadata, None)

    metadata_test = [('a', 'aval'), ('b', 'bval')]
    md.add_metadata(metadata_test)
    assert_equal(md.data, dict(metadata_test))

    md.add_metadata([['a', 'aval'], ['b', 'bval']])
    assert_equal(md.data, dict(metadata_test))

    md.add_metadata({'a': 'aval', 'b': 'bval'})
    assert_equal(md.data, dict(metadata_test))

    md.add_metadata({'a': 'aval', 'd': 'dval'})
    assert_equal(md.data, dict(metadata_test + [('d', 'dval')]))

    md.remove_metadata({'a': 'aval', 'd': 'dval'})
    assert_equal(md.data, dict(metadata_test[1:]))

    assert_raises(KeyError, md.remove_metadata, {'a': 'aval', 'd': 'dval'})
    assert_equal(md.to_xml().decode('utf-8'),
                 '<MetaData><MD><Name>b</Name><Value>bval</Value></MD></MetaData>')

    md.add_metadata({'b': 'cval'})
    assert_equal(md.data, dict([('b', 'cval')]))

    md.remove_metadata(['b', 'cval'])
    assert_equal(len(md.data), 0)
    assert_equal(md.to_xml().decode('utf-8'), '<MetaData />')

def test_cifti2_LabelTable():
    lt = ci.Cifti2LabelTable()
    assert_equal(len(lt), 0)
    assert_equal(len(lt.get_labels_as_dict()), 0)
    assert_raises(ci.CIFTI2HeaderError, lt.to_xml)
    label = ci.Cifti2Label(label='Test', key=0)
    lt.labels = [label]
    assert_equal(len(lt), 1)
    assert_equal(lt.get_labels_as_dict(), {label.key: label.label})

def test_cifti2_Label():
    lb = ci.Cifti2Label()
    lb.label = 'Test'
    lb.key = 0
    assert_equal(lb.rgba, (None, None, None, None))
    assert(compare_xml_leaf(
        lb.to_xml().decode('utf-8'),
        "<Label key='0'>Test</Label>"
    ))

    lb.red = 0
    lb.green = 0.1
    lb.blue = 0.2
    lb.alpha = 0.3
    assert_equal(lb.rgba, (0, 0.1, 0.2, 0.3))

    assert(compare_xml_leaf(
        lb.to_xml().decode('utf-8'),
        "<Label Key='0' Red='0' Green='0.1' Blue='0.2' Alpha='0.3'>Test</Label>"
    ))

def test_cifti2_parcel():
    pl = ci.Cifti2Parcel()
    assert_raises(ci.CIFTI2HeaderError, pl.to_xml)
    assert_raises(TypeError, pl.add_cifti_vertices, None)

def test_cifti2_voxelindicesijk():
    vi = ci.Cifti2VoxelIndicesIJK()
    assert_raises(ci.CIFTI2HeaderError, vi.to_xml)

def test_cifti2_vertices():
    vs = ci.Cifti2Vertices()
    assert_raises(ci.CIFTI2HeaderError, vs.to_xml)
    vs.brain_structure = 'CIFTI_STRUCTURE_OTHER'
    assert_equal(
        vs.to_xml().decode('utf-8'),
        '<Vertices BrainStructure="CIFTI_STRUCTURE_OTHER" />'
    )
    vs.vertices = np.array([0, 1, 2])
    assert_equal(
        vs.to_xml().decode('utf-8'),
        '<Vertices BrainStructure="CIFTI_STRUCTURE_OTHER">0 1 2</Vertices>'
    )

def test_cifti2_transformationmatrixvoxelindicesijktoxyz():
    tr = ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ()
    assert_raises(ci.CIFTI2HeaderError, tr.to_xml)

def test_cifti2_volume():
    vo = ci.Cifti2Volume()
    assert_raises(ci.CIFTI2HeaderError, vo.to_xml)

def test_cifti2_vertexindices():
    vi = ci.Cifti2VertexIndices()
    assert_raises(ci.CIFTI2HeaderError, vi.to_xml)
    vi.indices = np.array([0, 1, 2])
    assert_equal(
        vi.to_xml().decode('utf-8'),
        '<VertexIndices>0 1 2</VertexIndices>'
    )

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
