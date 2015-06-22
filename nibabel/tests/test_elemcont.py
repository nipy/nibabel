""" Testing element containers
"""
from __future__ import print_function

from ..elemcont import Elem, ElemDict, ElemList, InvalidElemError
from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)


def test_elemdict():
    # Test ElemDict class
    e = ElemDict()
    assert_raises(InvalidElemError, e.__setitem__, 'some', 'thing')
    assert_equal(list(e.keys()), [])
    elem = Elem('thing')
    e['some'] = elem
    assert_equal(list(e.keys()), ['some'])
    assert_equal(e['some'], 'thing')
    assert_equal(e.get_elem('some'), elem)

    # Test constructor
    assert_raises(InvalidElemError, ElemDict, dict(some='thing'))
    e = ElemDict(dict(some=Elem('thing')))
    assert_equal(list(e.keys()), ['some'])
    assert_equal(e['some'], 'thing')
    e = ElemDict(some=Elem('thing'))
    assert_equal(list(e.keys()), ['some'])
    assert_equal(e['some'], 'thing')


def test_elemlist():
    # Test ElemList class
    el = ElemList()
    assert_equal(len(el), 0)
    assert_raises(InvalidElemError, el.append, 'something')
    elem = Elem('something')
    el.append(elem)
    assert_equal(len(el), 1)
    assert_equal(el[0], 'something')
    assert_equal(el.get_elem(0), elem)
    assert_equal([x for x in el], ['something'])

    # Test constructor
    assert_raises(InvalidElemError, ElemList, ['something'])
    el = ElemList([elem])
    assert_equal(len(el), 1)
    assert_equal(el[0], 'something')
    assert_equal(el.get_elem(0), elem)
