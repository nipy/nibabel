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
    e2 = ElemDict(e)
    assert_equal(list(e2.keys()), ['some'])
    assert_equal(e2['some'], 'thing')


def test_elemdict_update():
    e1 = ElemDict(dict(some=Elem('thing')))
    e1.update(dict(hello=Elem('world')))
    assert_equal(list(e1.items()), [('some', 'thing'), ('hello', 'world')])
    e1 = ElemDict(dict(some=Elem('thing')))
    e2 = ElemDict(dict(hello=Elem('world')))
    e1.update(e2)
    assert_equal(list(e1.items()), [('some', 'thing'), ('hello', 'world')])


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
    el2 = ElemList(el)
    assert_equal(len(el2), 1)
    assert_equal(el2[0], 'something')
    assert_equal(el2.get_elem(0), elem)


def test_elemlist_slicing():
    el = ElemList()
    el[5:6] = [Elem('hello'), Elem('there'), Elem('world')]
    assert_equal([x for x in el], ['hello', 'there', 'world'])
    assert_true(isinstance(el[:2], ElemList))
    assert_equal([x for x in el[:2]], ['hello', 'there'])


def test_elemlist_add():
    res = ElemList([Elem('hello'), Elem('there')]) + ElemList([Elem('world')])
    assert_true(isinstance(res, ElemList))
    assert_equal([x for x in res], ['hello', 'there', 'world'])
    res = ElemList([Elem('hello'), Elem('there')]) + [Elem('world')]
    assert_true(isinstance(res, ElemList))
    assert_equal([x for x in res], ['hello', 'there', 'world'])
    res = [Elem('hello'), Elem('there')] + ElemList([Elem('world')])
    assert_true(isinstance(res, ElemList))
    assert_equal([x for x in res], ['hello', 'there', 'world'])
    res = ElemList([Elem('hello'), Elem('there')])
    res += [Elem('world')]
    assert_equal([x for x in res], ['hello', 'there', 'world'])
    res = ElemList([Elem('hello'), Elem('there')])
    res += ElemList([Elem('world')])
    assert_equal([x for x in res], ['hello', 'there', 'world'])