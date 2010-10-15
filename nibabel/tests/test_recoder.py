# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Tests recoder class '''

from nose.tools import assert_equal, assert_raises

from ..volumeutils import Recoder

def test_recoder():
    # simplest case, no aliases
    codes = ((1,),(2,))
    rc = Recoder(codes)
    yield assert_equal, rc.code[1], 1
    yield assert_equal, rc.code[2], 2
    yield assert_raises, KeyError, rc.code.__getitem__, 3
    # with explicit name for code
    rc = Recoder(codes, ['code1'])
    yield assert_raises, AttributeError, rc.__getattribute__, 'code'
    yield assert_equal, rc.code1[1], 1
    yield assert_equal, rc.code1[2], 2
    # code and label
    codes = ((1,'one'), (2,'two'))
    rc = Recoder(codes) # just with implicit alias
    yield assert_equal, rc.code[1], 1
    yield assert_equal, rc.code[2], 2
    yield assert_raises, KeyError, rc.code.__getitem__, 3
    yield assert_equal, rc.code['one'], 1
    yield assert_equal, rc.code['two'], 2
    yield assert_raises, KeyError, rc.code.__getitem__, 'three'
    yield assert_raises, AttributeError, rc.__getattribute__, 'label'
    rc = Recoder(codes, ['code1', 'label']) # with explicit column names
    yield assert_raises, AttributeError, rc.__getattribute__, 'code'
    yield assert_equal, rc.code1[1], 1
    yield assert_equal, rc.code1['one'], 1
    yield assert_equal, rc.label[1], 'one'
    yield assert_equal, rc.label['one'], 'one'
    # code, label, aliases
    codes = ((1,'one','1','first'), (2,'two'))
    rc = Recoder(codes) # just with implicit alias
    yield assert_equal, rc.code[1], 1
    yield assert_equal, rc.code['one'], 1
    yield assert_equal, rc.code['first'], 1
    rc = Recoder(codes, ['code1', 'label']) # with explicit column names
    yield assert_equal, rc.code1[1], 1
    yield assert_equal, rc.code1['first'], 1
    yield assert_equal, rc.label[1], 'one'
    yield assert_equal, rc.label['first'], 'one'
    # Don't allow funny names
    yield assert_raises, KeyError, Recoder, codes, ['field1']

def test_add_codes():
    codes = ((1,'one','1','first'), (2,'two'))
    rc = Recoder(codes)
    yield assert_equal, rc.code['two'], 2
    yield assert_raises, KeyError, rc.code.__getitem__, 'three'
    rc.add_codes(((3, 'three'), (1, 'number 1')))
    yield assert_equal, rc.code['three'], 3
    yield assert_equal, rc.code['number 1'], 1

def test_sugar():
    # Syntactic sugar for recoder class
    codes = ((1,'one','1','first'), (2,'two'))
    rc = Recoder(codes)
    # Field1 is synonym for first named dict
    yield assert_equal, rc.code, rc.field1
    rc = Recoder(codes, fields=('code1', 'label'))
    yield assert_equal, rc.code1, rc.field1
    # Direct key access identical to key access for first named 
    yield assert_equal, rc[1], rc.field1[1]
    yield assert_equal, rc['two'], rc.field1['two']
    # keys gets all keys
    yield assert_equal, set(rc.keys()), set((1,'one','1','first',2,'two'))
    # value_set gets set of values from first column
    yield assert_equal, rc.value_set(), set((1, 2))
    # or named column if given
    yield assert_equal, rc.value_set('label'), set(('one', 'two'))
