""" Test kw_only decorators """

from ..keywordonly import kw_only_func, kw_only_meth

from nose.tools import assert_equal
from nose.tools import assert_raises


def test_kw_only_func():
    # Test decorator
    def func(an_arg):
        "My docstring"
        return an_arg
    assert_equal(func(1), 1)
    assert_raises(TypeError, func, 1, 2)
    dec_func = kw_only_func(1)(func)
    assert_equal(dec_func(1), 1)
    assert_raises(TypeError, dec_func, 1, 2)
    assert_raises(TypeError, dec_func, 1, akeyarg=3)
    assert_equal(dec_func.__doc__, 'My docstring')

    @kw_only_func(1)
    def kw_func(an_arg, a_kwarg='thing'):
        "Another docstring"
        return an_arg, a_kwarg
    assert_equal(kw_func(1), (1, 'thing'))
    assert_raises(TypeError, kw_func, 1, 2)
    assert_equal(kw_func(1, a_kwarg=2), (1, 2))
    assert_raises(TypeError, kw_func, 1, akeyarg=3)
    assert_equal(kw_func.__doc__, 'Another docstring')

    class C(object):

        @kw_only_meth(1)
        def kw_meth(self, an_arg, a_kwarg='thing'):
            "Method docstring"
            return an_arg, a_kwarg
    c = C()
    assert_equal(c.kw_meth(1), (1, 'thing'))
    assert_raises(TypeError, c.kw_meth, 1, 2)
    assert_equal(c.kw_meth(1, a_kwarg=2), (1, 2))
    assert_raises(TypeError, c.kw_meth, 1, akeyarg=3)
    assert_equal(c.kw_meth.__doc__, 'Method docstring')
