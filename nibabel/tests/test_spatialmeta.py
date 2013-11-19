# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests for spatialmeta"""

import sys
from os import path
from glob import glob
from collections import OrderedDict
import numpy as np

from nose.tools import ok_, eq_, assert_raises
from unittest import TestCase

from .. import spatialmeta

def test_is_constant():
    ok_(spatialmeta.is_constant([0]))
    ok_(spatialmeta.is_constant([0, 0]))
    ok_(spatialmeta.is_constant([0, 0, 1, 1], period=2))
    ok_(spatialmeta.is_constant([0, 0, 1, 1, 2, 2], period=2))
    eq_(spatialmeta.is_constant([0, 1]), False)
    eq_(spatialmeta.is_constant([0, 0, 1, 2], 2), False)
    assert_raises(ValueError, spatialmeta.is_constant, [0, 0, 0], -1)
    assert_raises(ValueError, spatialmeta.is_constant, [0, 0, 0], 1)
    assert_raises(ValueError, spatialmeta.is_constant, [0, 0, 0], 2)
    assert_raises(ValueError, spatialmeta.is_constant, [0, 0, 0], 4)
    
def test_is_repeating():
    ok_(spatialmeta.is_repeating([0, 1, 0, 1], 2))
    ok_(spatialmeta.is_repeating([0, 1, 0, 1, 0, 1], 2))
    eq_(spatialmeta.is_repeating([0, 1, 1, 2], 2), False)
    assert_raises(ValueError, spatialmeta.is_repeating, [0, 1, 0, 1], -1)
    assert_raises(ValueError, spatialmeta.is_repeating, [0, 1, 0, 1], 1)
    assert_raises(ValueError, spatialmeta.is_repeating, [0, 1, 0, 1], 3)
    assert_raises(ValueError, spatialmeta.is_repeating, [0, 1, 0, 1], 4)
    assert_raises(ValueError, spatialmeta.is_repeating, [0, 1, 0, 1], 5)

def test_properties():
    sm = spatialmeta.SpatialMeta((2, 2, 2), np.eye(4))
    eq_(sm.shape, (2, 2, 2))
    ok_(np.allclose(sm.affine, np.eye(4)))
    eq_(sm.version, spatialmeta._meta_version)
    
    sm.shape = (3, 3, 3)
    eq_(sm.shape, (3, 3, 3))
    
    sm.affine = np.diag([1, 2, 3, 1])
    ok_(np.allclose(sm.affine, np.diag([1, 2, 3, 1])))
    
def test_get_classes():
    sm = spatialmeta.SpatialMeta((2, 2, 2), np.eye(4))
    eq_(set(sm.get_classes()), 
        set(('varies_over()', 
             'varies_over(0)',
             'varies_over(1)',
             'varies_over(2)',
            )
           )
       )
    
    sm.shape = (2, 2, 2, 2)
    eq_(set(sm.get_classes()), 
        set(('varies_over()', 
             'varies_over(0)',
             'varies_over(1)',
             'varies_over(2)',
             'varies_over(3)',
             'varies_over(0,3)',
             'varies_over(1,3)',
             'varies_over(2,3)',
            )
           )
       )
       
    sm.shape = (2, 2, 2, 1, 2)
    eq_(set(sm.get_classes()), 
        set(('varies_over()', 
             'varies_over(0)',
             'varies_over(1)',
             'varies_over(2)',
             'varies_over(4)',
             'varies_over(0,4)',
             'varies_over(1,4)',
             'varies_over(2,4)',
            )
           )
       )
 
    sm.shape = (2, 2, 2, 2, 2)
    eq_(set(sm.get_classes()), 
        set(('varies_over()', 
             'varies_over(0)',
             'varies_over(1)',
             'varies_over(2)',
             'varies_over(3)',
             'varies_over(4)',
             'varies_over(0,3)',
             'varies_over(1,3)',
             'varies_over(2,3)',
             'varies_over(0,4)',
             'varies_over(1,4)',
             'varies_over(2,4)',
             'varies_over(3,4)',
             'varies_over(0,3,4)',
             'varies_over(1,3,4)',
             'varies_over(2,3,4)',
            )
           )
       )

def test_get_varying_dims():
    eq_(spatialmeta.SpatialMeta.get_varying_dims('varies_over()'), [])
    eq_(spatialmeta.SpatialMeta.get_varying_dims('varies_over(0)'), [0])
    eq_(spatialmeta.SpatialMeta.get_varying_dims('varies_over(1,2)'), [1,2])
    eq_(spatialmeta.SpatialMeta.get_varying_dims('varies_over(2,3,4)'), 
        [2,3,4]
       )
       
def test_get_n_vals():
    sm = spatialmeta.SpatialMeta((2, 3, 5, 7, 9), np.eye(4))
    eq_(sm.get_n_vals('varies_over()'), 1)
    eq_(sm.get_n_vals('varies_over(0)'), 2)
    eq_(sm.get_n_vals('varies_over(1)'), 3)
    eq_(sm.get_n_vals('varies_over(2)'), 5)
    eq_(sm.get_n_vals('varies_over(3)'), 7)
    eq_(sm.get_n_vals('varies_over(4)'), 9)
    eq_(sm.get_n_vals('varies_over(3,4)'), 7 * 9)
    eq_(sm.get_n_vals('varies_over(0,3)'), 2 * 7)
    eq_(sm.get_n_vals('varies_over(1,3)'), 3 * 7)
    eq_(sm.get_n_vals('varies_over(2,3)'), 5 * 7)
    eq_(sm.get_n_vals('varies_over(0,4)'), 2 * 9)
    eq_(sm.get_n_vals('varies_over(1,4)'), 3 * 9)
    eq_(sm.get_n_vals('varies_over(2,4)'), 5 * 9)
    eq_(sm.get_n_vals('varies_over(0,3,4)'), 2 * 7 * 9)
    eq_(sm.get_n_vals('varies_over(1,3,4)'), 3 * 7 * 9)
    eq_(sm.get_n_vals('varies_over(2,3,4)'), 5 * 7 * 9)

class TestNestedElements(TestCase):
    def setUp(self):
        self.sm = spatialmeta.SpatialMeta((2,2,2), np.eye(4))
        self.sm.set_nested('varies_over()', ('test1',), 'foo')
        self.sm.set_nested('varies_over()', ('test2', 'subtest1'), 'bar')
        self.sm.set_nested('varies_over()', ('test2', 'subtest2'), [2, 1, 0])
        self.sm.set_nested('varies_over()', ('test2', 'subtest2', 1), 3)
        self.sm.set_nested('varies_over()', 
                            ('test2', 'subtest3'), 
                            {'subsubtest1' : 'baz'}
                           )
        self.sm.set_nested('varies_over(0)', ('test3',), 'foobar')
    
    def test_get_set_nested(self):
        eq_(self.sm.get_nested('varies_over()', ('test1',)), 'foo')
        eq_(self.sm.get_nested('varies_over()', ('test2', 'subtest1')), 
            'bar'
           )
        eq_(self.sm.get_nested('varies_over()', ('test2', 'subtest2')), 
            [2, 3, 0]
           )
        eq_(self.sm.get_nested('varies_over()', ('test2', 'subtest2', 0)), 2)
        eq_(self.sm.get_nested('varies_over()', ('test2', 'subtest3')), 
            {'subsubtest1' : 'baz'}
           )
        eq_(self.sm.get_nested('varies_over()', 
                                ('test2', 'subtest3', 'subsubtest1')),
            'baz'
           ) 
        eq_(self.sm.get_nested('varies_over(0)', ('test3',)), 'foobar')
        
        assert_raises(KeyError, 
                      self.sm.get_nested, 
                      'varies_over()', 
                      ('test1', 'blah')
                     )
        assert_raises(KeyError, 
                      self.sm.get_nested, 
                      'varies_over()', 
                      ('test3',)
                     )
        
    def test_contains_nested(self):
        ok_(self.sm.contains_nested('varies_over()', ('test1',)))
        ok_(self.sm.contains_nested('varies_over()', ('test2',)))
        ok_(self.sm.contains_nested('varies_over()', ('test2', 'subtest1')))
        eq_(self.sm.contains_nested('varies_over(0)', ('test1',)), False)
        
    def test_remove_nested(self):
        assert_raises(KeyError, 
                      self.sm.remove_nested, 
                      'varies_over()', 
                      ('test1', 'blah')
                     )
        assert_raises(KeyError, 
                      self.sm.remove_nested, 
                      'varies_over()', 
                      ('blah',)
                     )
        
        self.sm.remove_nested('varies_over()', ('test1',))
        assert_raises(KeyError, 
                      self.sm.get_nested, 
                      'varies_over()', 
                      ('test1',)
                     )
        self.sm.remove_nested('varies_over()', ('test2', 'subtest1'))
        eq_(self.sm.get_nested('varies_over()', ('test2', 'subtest2')), 
            [2, 3, 0]
           )
        self.sm.remove_nested('varies_over()', ('test2', 'subtest2'))
        self.sm.remove_nested('varies_over()', ('test2', 'subtest3'))
        assert_raises(KeyError, 
                      self.sm.get_nested, 
                      'varies_over()', 
                      ('test2',)
                     )

    def test_iter_elements(self):
        eq_(list(self.sm.iter_elements('varies_over()')),
            [(('test1',), 'foo'),
             (('test2', 'subtest1'), 'bar'),
             (('test2', 'subtest2'), [2, 3, 0]),
             (('test2', 'subtest3', 'subsubtest1'), 'baz'),
            ]
           )
        eq_(list(self.sm.iter_elements('varies_over(0)')), 
            [(('test3',), 'foobar')]
           )
        eq_(list(self.sm.iter_elements('varies_over(1)')), 
            []
           )
    
def test_check_valid():
    sm = spatialmeta.SpatialMeta((2,3,5), np.eye(4))
    sm.check_valid()
    
    sm.set_nested('varies_over(0)', ('test1',), ['bar'])
    assert_raises(spatialmeta.InvalidSpatialMetaError,
                  sm.check_valid,
                 )
    sm.set_nested('varies_over(0)', ('test1',), ['foo', 'bar'])
    sm.check_valid()
    
    sm.set_nested('varies_over()', ('test1',), 'foo')
    assert_raises(spatialmeta.InvalidSpatialMetaError,
                  sm.check_valid,
                 )
                 
def test_get_values_and_class():
    sm = spatialmeta.SpatialMeta((2,3,5), np.eye(4))
    sm.set_nested('varies_over()', ('test1',), 'foo')
    sm.set_nested('varies_over(0)', ('test2',), ['foo', 'bar'])
    sm.set_nested('varies_over(1)', ('test3',), ['foo', 'bar', 'baz'])
    sm.set_nested('varies_over(2)', ('test4',), [0, 1, 2, 3, 4])
    sm.set_nested('varies_over()', ('test5',), {'subtest1' : 'foo'})
    
    eq_(sm.get_values_and_class(('test1',)), 
        ('foo', 'varies_over()')
       )
    eq_(sm.get_values_and_class(('test2',)), 
        (['foo', 'bar'], 'varies_over(0)')
       )
    eq_(sm.get_values_and_class(('test3',)), 
        (['foo', 'bar', 'baz'], 'varies_over(1)')
       )
    eq_(sm.get_values_and_class(('test4',)), 
        ([0, 1, 2, 3, 4], 'varies_over(2)')
       )
    eq_(sm.get_values_and_class(('test5', 'subtest1')), 
        ('foo', 'varies_over()')
       )
    eq_(sm.get_values_and_class(('blah',)), 
        (None, None)
       )
    
    assert_raises(ValueError, sm.get_values_and_class, ('test5',))
       
def test_get_value():
    sm = spatialmeta.SpatialMeta((2,3,5,7,11), np.eye(4))
    sm.set_nested('varies_over()', ('test1',), 'foo')
    sm.set_nested('varies_over(0)', ('test2',), ['foo', 'bar'])
    sm.set_nested('varies_over(1)', ('test3',), ['foo', 'bar', 'baz'])
    sm.set_nested('varies_over(2)', ('test4',), range(5))
    sm.set_nested('varies_over(3)', ('test5',), range(7))
    sm.set_nested('varies_over(4)', ('test6',), range(11))
    sm.set_nested('varies_over(0,3)', ('test7',), range(14))
    sm.set_nested('varies_over(0,4)', ('test8',), range(22))
    sm.set_nested('varies_over(0,3,4)', ('test9',), range(154))
    
    assert_raises(IndexError, sm.get_value, ('test1',), (0, 0, 0))
    assert_raises(IndexError, sm.get_value, ('test1',), (0, 0, 0, 0, 0, 0))
    assert_raises(IndexError, sm.get_value, ('test1',), (0, 0, 0, 0, -1))
    assert_raises(IndexError, sm.get_value, ('test1',), (2, 0, 0, 0, 0))
    
    eq_(sm.get_value(('test1',), (0, 0, 0, 0, 0)), 'foo')
    eq_(sm.get_value(('test2',), (0, 0, 0, 0, 0)), 'foo')
    eq_(sm.get_value(('test2',), (1, 0, 0, 0, 0)), 'bar')
    eq_(sm.get_value(('test3',), (1, 0, 0, 0, 0)), 'foo')
    eq_(sm.get_value(('test3',), (1, 1, 0, 0, 0)), 'bar')
    eq_(sm.get_value(('test3',), (1, 2, 0, 0, 0)), 'baz')
    eq_(sm.get_value(('test4',), (1, 2, 0, 0, 0)), 0)
    eq_(sm.get_value(('test4',), (1, 2, 2, 0, 0)), 2)
    eq_(sm.get_value(('test5',), (1, 2, 2, 0, 0)), 0)
    eq_(sm.get_value(('test5',), (1, 2, 2, 3, 0)), 3)
    eq_(sm.get_value(('test6',), (1, 2, 2, 3, 0)), 0)
    eq_(sm.get_value(('test6',), (1, 2, 2, 3, 9)), 9)
    eq_(sm.get_value(('test7',), (0, 2, 3, 0, 5)), 0)
    eq_(sm.get_value(('test7',), (1, 2, 3, 0, 5)), 1)
    eq_(sm.get_value(('test7',), (1, 2, 3, 2, 5)), 5)
    eq_(sm.get_value(('test8',), (0, 2, 3, 2, 0)), 0)
    eq_(sm.get_value(('test8',), (1, 2, 3, 2, 3)), 7)
    eq_(sm.get_value(('test9',), (0, 2, 3, 0, 0)), 0)
    eq_(sm.get_value(('test9',), (0, 2, 3, 2, 3)), 46)
    eq_(sm.get_value(('test9',), (1, 2, 3, 2, 3)), 47)
    
    
