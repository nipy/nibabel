""" Testing deprecator module / Deprecator class
"""

import sys
import warnings
from functools import partial

from nose.tools import (assert_true, assert_raises, assert_equal)

from nibabel.deprecator import (_ensure_cr, _add_dep_doc,
                                ExpiredDeprecationError, Deprecator)

from ..testing import clear_and_catch_warnings

_OWN_MODULE = sys.modules[__name__]


def test__ensure_cr():
    # Make sure text ends with carriage return
    assert_equal(_ensure_cr('  foo'), '  foo\n')
    assert_equal(_ensure_cr('  foo\n'), '  foo\n')
    assert_equal(_ensure_cr('  foo  '), '  foo\n')
    assert_equal(_ensure_cr('foo  '), 'foo\n')
    assert_equal(_ensure_cr('foo  \n bar'), 'foo  \n bar\n')
    assert_equal(_ensure_cr('foo  \n\n'), 'foo\n')


def test__add_dep_doc():
    # Test utility function to add deprecation message to docstring
    assert_equal(_add_dep_doc('', 'foo'), 'foo\n')
    assert_equal(_add_dep_doc('bar', 'foo'), 'bar\n\nfoo\n')
    assert_equal(_add_dep_doc('   bar', 'foo'), '   bar\n\nfoo\n')
    assert_equal(_add_dep_doc('   bar', 'foo\n'), '   bar\n\nfoo\n')
    assert_equal(_add_dep_doc('bar\n\n', 'foo'), 'bar\n\nfoo\n')
    assert_equal(_add_dep_doc('bar\n    \n', 'foo'), 'bar\n\nfoo\n')
    assert_equal(_add_dep_doc(' bar\n\nSome explanation', 'foo\nbaz'),
                 ' bar\n\nfoo\nbaz\n\nSome explanation\n')
    assert_equal(_add_dep_doc(' bar\n\n  Some explanation', 'foo\nbaz'),
                 ' bar\n  \n  foo\n  baz\n  \n  Some explanation\n')


class CustomError(Exception):
    """ Custom error class for testing expired deprecation errors """


def cmp_func(v):
    """ Comparison func tests against version 2.0 """
    return (float(v) > 2) - (float(v) < 2)


def func_no_doc():
    pass


def func_doc(i):
    "A docstring"


def func_doc_long(i, j):
    "A docstring\n\n   Some text"


class TestDeprecatorFunc(object):
    """ Test deprecator function specified in ``dep_func`` """

    dep_func = Deprecator(cmp_func)

    def test_dep_func(self):
        # Test function deprecation
        dec = self.dep_func
        func = dec('foo')(func_no_doc)
        with clear_and_catch_warnings(modules=[_OWN_MODULE]) as w:
            warnings.simplefilter('always')
            assert_equal(func(), None)
            assert_equal(len(w), 1)
            assert_true(w[0].category is DeprecationWarning)
        assert_equal(func.__doc__, 'foo\n')
        func = dec('foo')(func_doc)
        with clear_and_catch_warnings(modules=[_OWN_MODULE]) as w:
            warnings.simplefilter('always')
            assert_equal(func(1), None)
            assert_equal(len(w), 1)
        assert_equal(func.__doc__, 'A docstring\n\nfoo\n')
        func = dec('foo')(func_doc_long)
        with clear_and_catch_warnings(modules=[_OWN_MODULE]) as w:
            warnings.simplefilter('always')
            assert_equal(func(1, 2), None)
            assert_equal(len(w), 1)
        assert_equal(func.__doc__, 'A docstring\n   \n   foo\n   \n   Some text\n')

        # Try some since and until versions
        func = dec('foo', '1.1')(func_no_doc)
        assert_equal(func.__doc__, 'foo\n\n* deprecated from version: 1.1\n')
        with clear_and_catch_warnings(modules=[_OWN_MODULE]) as w:
            warnings.simplefilter('always')
            assert_equal(func(), None)
            assert_equal(len(w), 1)
        func = dec('foo', until='2.4')(func_no_doc)
        with clear_and_catch_warnings(modules=[_OWN_MODULE]) as w:
            warnings.simplefilter('always')
            assert_equal(func(), None)
            assert_equal(len(w), 1)
        assert_equal(func.__doc__,
                    'foo\n\n* Will raise {} as of version: 2.4\n'
                    .format(ExpiredDeprecationError))
        func = dec('foo', until='1.8')(func_no_doc)
        assert_raises(ExpiredDeprecationError, func)
        assert_equal(func.__doc__,
                    'foo\n\n* Raises {} as of version: 1.8\n'
                    .format(ExpiredDeprecationError))
        func = dec('foo', '1.2', '1.8')(func_no_doc)
        assert_raises(ExpiredDeprecationError, func)
        assert_equal(func.__doc__,
                    'foo\n\n* deprecated from version: 1.2\n'
                    '* Raises {} as of version: 1.8\n'
                    .format(ExpiredDeprecationError))
        func = dec('foo', '1.2', '1.8')(func_doc_long)
        assert_equal(func.__doc__,
                    'A docstring\n   \n   foo\n   \n'
                    '   * deprecated from version: 1.2\n'
                    '   * Raises {} as of version: 1.8\n   \n'
                    '   Some text\n'
                    .format(ExpiredDeprecationError))
        assert_raises(ExpiredDeprecationError, func)

        # Check different warnings and errors
        func = dec('foo', warn_class=UserWarning)(func_no_doc)
        with clear_and_catch_warnings(modules=[_OWN_MODULE]) as w:
            warnings.simplefilter('always')
            assert_equal(func(), None)
            assert_equal(len(w), 1)
            assert_true(w[0].category is UserWarning)

        func = dec('foo', error_class=CustomError)(func_no_doc)
        with clear_and_catch_warnings(modules=[_OWN_MODULE]) as w:
            warnings.simplefilter('always')
            assert_equal(func(), None)
            assert_equal(len(w), 1)
            assert_true(w[0].category is DeprecationWarning)

        func = dec('foo', until='1.8', error_class=CustomError)(func_no_doc)
        assert_raises(CustomError, func)


class TestDeprecatorMaker(object):
    """ Test deprecator class creation with custom warnings and errors """

    dep_maker = partial(Deprecator, cmp_func)

    def test_deprecator_maker(self):
        dec = self.dep_maker(warn_class=UserWarning)
        func = dec('foo')(func_no_doc)
        with clear_and_catch_warnings(modules=[_OWN_MODULE]) as w:
            warnings.simplefilter('always')
            assert_equal(func(), None)
            assert_equal(len(w), 1)
            assert_true(w[0].category is UserWarning)

        dec = self.dep_maker(error_class=CustomError)
        func = dec('foo')(func_no_doc)
        with clear_and_catch_warnings(modules=[_OWN_MODULE]) as w:
            warnings.simplefilter('always')
            assert_equal(func(), None)
            assert_equal(len(w), 1)
            assert_true(w[0].category is DeprecationWarning)

        func = dec('foo', until='1.8')(func_no_doc)
        assert_raises(CustomError, func)
