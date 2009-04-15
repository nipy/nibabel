''' Utilities to change context for nose.tools tests '''

import nose.tools as nt

assert_equal = lambda x, y: nt.assert_equal(x, y)
assert_not_equal = lambda x, y: nt.assert_not_equal(x, y)
assert_true = lambda x: nt.assert_true(x)
assert_false = lambda x: nt.assert_false(x)

def assert_raises(error, func, *args, **kwargs):
    return nt.assert_raises(error, func, *args, **kwargs)
