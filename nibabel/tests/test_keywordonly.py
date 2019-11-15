""" Test kw_only decorators """

from ..keywordonly import kw_only_func, kw_only_meth

import pytest


def test_kw_only_func():
    # Test decorator
    def func(an_arg):
        "My docstring"
        return an_arg
    assert func(1) == 1
    with pytest.raises(TypeError):
        func(1, 2)
    dec_func = kw_only_func(1)(func)
    assert dec_func(1) == 1
    with pytest.raises(TypeError):
        dec_func(1, 2)
    with pytest.raises(TypeError):
        dec_func(1, akeyarg=3)
    assert dec_func.__doc__ == 'My docstring'

    @kw_only_func(1)
    def kw_func(an_arg, a_kwarg='thing'):
        "Another docstring"
        return an_arg, a_kwarg
    assert kw_func(1) == (1, 'thing')
    with pytest.raises(TypeError):
        kw_func(1, 2)
    assert kw_func(1, a_kwarg=2) == (1, 2)
    with pytest.raises(TypeError):
        kw_func(1, akeyarg=3)
    assert kw_func.__doc__ == 'Another docstring'

    class C(object):

        @kw_only_meth(1)
        def kw_meth(self, an_arg, a_kwarg='thing'):
            "Method docstring"
            return an_arg, a_kwarg
    c = C()
    assert c.kw_meth(1) == (1, 'thing')
    with pytest.raises(TypeError):
        c.kw_meth(1, 2)
    assert c.kw_meth(1, a_kwarg=2) == (1, 2)
    with pytest.raises(TypeError):
        c.kw_meth(1, akeyarg=3)
    assert c.kw_meth.__doc__ == 'Method docstring'
