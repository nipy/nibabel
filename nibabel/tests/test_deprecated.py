""" Testing `deprecated` module
"""

import warnings

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)

from ..deprecated import ModuleProxy, FutureWarningMixin


def test_module_proxy():
    # Test proxy for module
    mp = ModuleProxy('nibabel.deprecated')
    assert_true(hasattr(mp, 'ModuleProxy'))
    assert_true(mp.ModuleProxy is ModuleProxy)
    assert_equal(repr(mp), '<module proxy for nibabel.deprecated>')


def test_futurewarning_mixin():
    # Test mixin for FutureWarning
    class C(object):
        def __init__(self, val):
            self.val = val
        def meth(self):
            return self.val
    class D(FutureWarningMixin, C):
        pass
    class E(FutureWarningMixin, C):
        warn_message = "Oh no, not this one"
    with warnings.catch_warnings(record=True) as warns:
        c = C(42)
        assert_equal(c.meth(), 42)
        assert_equal(warns, [])
        d = D(42)
        assert_equal(d.meth(), 42)
        warn = warns.pop(0)
        assert_equal(warn.category, FutureWarning)
        assert_equal(str(warn.message),
                     'This class will be removed in future versions')
        e = E(42)
        assert_equal(e.meth(), 42)
        warn = warns.pop(0)
        assert_equal(warn.category, FutureWarning)
        assert_equal(str(warn.message), 'Oh no, not this one')
