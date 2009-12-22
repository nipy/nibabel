''' Utilities to change context for nose.tools tests '''

from os.path import join as pjoin, split as psplit, abspath

import nose.tools as nt

example_data_path = abspath(pjoin(psplit(__file__)[0], '..', 'tests', 'data'))

assert_equal = lambda x, y: nt.assert_equal(x, y)
assert_not_equal = lambda x, y: nt.assert_not_equal(x, y)
assert_true = lambda x: nt.assert_true(x)
assert_false = lambda x: nt.assert_false(x)

def assert_raises(error, func, *args, **kwargs):
    return nt.assert_raises(error, func, *args, **kwargs)
