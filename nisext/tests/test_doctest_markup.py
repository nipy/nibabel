""" Testing doctest markup tests
"""

from ..py3builder import doctest_markup

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_true, assert_equal, assert_raises


def test_search_replace():
    # Test search and replace regexps
    assert_equal(['any string'], doctest_markup(['any string']))
    line = 'from io import StringIO as BytesIO'
    assert_equal([line], doctest_markup([line]))
    line = '>>> ' + line
    assert_equal(['>>> from io import BytesIO'], doctest_markup([line]))
    line = line + '  '
    assert_equal(['>>> from io import BytesIO'], doctest_markup([line]))
    line = line + '\n'
    assert_equal(['>>> from io import BytesIO\n'], doctest_markup([line]))
    # Bytes output
    assert_equal(['any', '  some'], doctest_markup(['any', '  some']))
    assert_equal(['any #23: bytes', '  some'],
                 doctest_markup(['any #23: bytes', '  some']))
    assert_equal([' >>> any #23: bytes \n', '  bsome \n'],
                 doctest_markup([' >>> any #23: bytes \n', '  some \n']))
