""" Testing gifti objects
"""

import numpy as np

from ...nifti1 import data_type_codes, intent_codes

from ... import cifti2 as ci

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_equal, assert_raises


def test_cifti2_metadata():
    md = ci.Cifti2MetaData()
    assert_equal(md.data, [])
    assert_raises(ValueError, md.add_metadata, 0)
    assert_raises(ValueError, md.add_metadata, ['a'])
    assert_raises(ValueError, md.remove_metadata, None)
    assert_raises(ValueError, md._add_remove_metadata, [('a', 'b')], 'loren')

    metadata_test = [('a', 'aval'), ('b', 'bval')]
    md.add_metadata(metadata_test)
    assert_equal(md.data, metadata_test)

    md.add_metadata([['a', 'aval'], ['b', 'bval']])
    assert_equal(md.data, metadata_test)

    md.add_metadata({'a': 'aval', 'b': 'bval'})
    assert_equal(md.data, metadata_test)

    md.add_metadata({'a': 'aval', 'd': 'dval'})
    assert_equal(md.data, metadata_test + [('d', 'dval')])

    md.remove_metadata({'a': 'aval', 'd': 'dval'})
    assert_equal(md.data, metadata_test[1:])

    assert_raises(ValueError, md.remove_metadata, {'a': 'aval', 'd': 'dval'})
    assert_equal(md.to_xml().decode('utf-8'),
                 '<MetaData><MD><Name>b</Name><Value>bval</Value></MD></MetaData>')

    md.remove_metadata(['b', 'bval'])
    assert_equal(len(md.data), 0)
    assert_equal(md.to_xml().decode('utf-8'), '<MetaData />')


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
