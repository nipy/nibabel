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
    assert_raises(ValueError, md.add_metadata, ['a'])

    md.add_metadata([('a', 'aval'), ('b', 'bval')])
    assert_equal(len(md.data), 2)

    md.add_metadata([['a', 'aval'], ['b', 'bval']])
    assert_equal(len(md.data), 2)

    md.add_metadata({'a': 'aval', 'b': 'bval'})
    assert_equal(len(md.data), 2)

    md.add_metadata({'a': 'aval', 'd': 'dval'})
    assert_equal(len(md.data), 3)

    md.remove_metadata({'a': 'aval', 'd': 'dval'})
    assert_equal(len(md.data), 1)

    assert_raises(ValueError, md.remove_metadata, {'a': 'aval', 'd': 'dval'})
    assert_equal(md.to_xml().decode('utf-8'),
                 '<MetaData><MD><Name>b</Name><Value>bval</Value></MD></MetaData>')

    md.remove_metadata(['b', 'bval'])
    assert_equal(len(md.data), 0)
    assert_equal(md.to_xml().decode('utf-8'), '<MetaData />')
