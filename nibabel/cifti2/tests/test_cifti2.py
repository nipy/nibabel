""" Testing gifti objects
"""
import collections
from lxml import etree

import numpy as np

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

def test_cifti2_LabelTable():
    lt = ci.Cifti2LabelTable()
    assert_equal(len(lt), 0)
    assert_raises(ci.CIFTI2HeaderError, lt.to_xml)

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



def test_cifti2_Label():
    lb = ci.Cifti2Label()
    lb.label = 'Test'
    lb.key = 0
    assert_equal(lb.rgba, (0, 0, 0, 0))
    assert(compare_xml_leaf(
        lb.to_xml().decode('utf-8'),
        "<Label Key='0' Red='0' Green='0' Blue='0' Alpha='0'>Test</Label>"
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

    lb.red = 10
    assert_raises(ci.CIFTI2HeaderError, lb.to_xml)
    lb.red = 0

    lb.key = 'a'
    assert_raises(ci.CIFTI2HeaderError, lb.to_xml)
    lb.key = 0

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
