''' Utilities for testing '''
from os.path import dirname, abspath, join as pjoin

# set path to example data
data_path = abspath(pjoin(dirname(__file__), '..', 'tests', 'data'))

# Allow failed import of nose if not now running tests
try:
    import nose.tools as nt
except ImportError:
    pass
else:
    from lightunit import ParametricTestCase, parametric
    # wrappers to change context for nose.tools tests '''
    assert_equal = lambda x, y: nt.assert_equal(x, y)
    assert_not_equal = lambda x, y: nt.assert_not_equal(x, y)
    assert_true = lambda x: nt.assert_true(x)
    assert_false = lambda x: nt.assert_false(x)

    def assert_raises(error, func, *args, **kwargs):
        return nt.assert_raises(error, func, *args, **kwargs)

