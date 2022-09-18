""" Testing element containers
"""
from __future__ import print_function

from ..elemcont import MetaElem, ElemDict, ElemList, InvalidElemError

import pytest


def test_elemdict():
    # Test ElemDict class
    e = ElemDict()
    with pytest.raises(InvalidElemError):
        e['some'] = 'thing'
    assert list(e.keys()) == []
    elem = MetaElem('thing')
    e['some'] = elem
    assert list(e.keys()) == ['some']
    assert e['some'] == 'thing'
    assert e.get_elem('some') == elem

    # Test constructor
    with pytest.raises(InvalidElemError):
        ElemDict(dict(some='thing'))
    e = ElemDict(dict(some=MetaElem('thing')))
    assert list(e.keys()) == ['some']
    assert e['some'] == 'thing'
    e = ElemDict(some=MetaElem('thing'))
    assert list(e.keys()) == ['some']
    assert e['some'] == 'thing'
    e2 = ElemDict(e)
    assert list(e2.keys()) == ['some']
    assert e2['some'] == 'thing'


def test_elemdict_update():
    e1 = ElemDict(dict(some=MetaElem('thing')))
    e1.update(dict(hello=MetaElem('world')))
    assert list(e1.items()) == [('some', 'thing'), ('hello', 'world')]
    e1 = ElemDict(dict(some=MetaElem('thing')))
    e2 = ElemDict(dict(hello=MetaElem('world')))
    e1.update(e2)
    assert list(e1.items()) == [('some', 'thing'), ('hello', 'world')]


def test_elemlist():
    # Test ElemList class
    el = ElemList()
    assert len(el) == 0
    with pytest.raises(InvalidElemError):
        el.append('something')
    elem = MetaElem('something')
    el.append(elem)
    assert len(el) == 1
    assert el[0] == 'something'
    assert el.get_elem(0) == elem
    assert [x for x in el] == ['something']

    # Test constructor
    with pytest.raises(InvalidElemError):
        ElemList(['something'])
    el = ElemList([elem])
    assert len(el) == 1
    assert el[0] == 'something'
    assert el.get_elem(0) == elem
    el2 = ElemList(el)
    assert len(el2) == 1
    assert el2[0] == 'something'
    assert el2.get_elem(0) == elem


def test_elemlist_slicing():
    el = ElemList()
    el[5:6] = [MetaElem('hello'), MetaElem('there'), MetaElem('world')]
    assert [x for x in el] == ['hello', 'there', 'world']
    assert isinstance(el[:2], ElemList)
    assert [x for x in el[:2]] == ['hello', 'there']


def test_elemlist_add():
    res = ElemList([MetaElem('hello'), MetaElem('there')]) + ElemList([MetaElem('world')])
    assert isinstance(res, ElemList)
    assert [x for x in res] == ['hello', 'there', 'world']
    res = ElemList([MetaElem('hello'), MetaElem('there')]) + [MetaElem('world')]
    assert isinstance(res, ElemList)
    assert [x for x in res] == ['hello', 'there', 'world']
    res = [MetaElem('hello'), MetaElem('there')] + ElemList([MetaElem('world')])
    assert isinstance(res, ElemList)
    assert [x for x in res] == ['hello', 'there', 'world']
    res = ElemList([MetaElem('hello'), MetaElem('there')])
    res += [MetaElem('world')]
    assert [x for x in res] == ['hello', 'there', 'world']
    res = ElemList([MetaElem('hello'), MetaElem('there')])
    res += ElemList([MetaElem('world')])
    assert [x for x in res] == ['hello', 'there', 'world']
