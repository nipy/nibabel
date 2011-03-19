""" Testing doctest markup tests
"""

from ..py3builder import doctest_markup, byter

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
    marked_lines = ['any', '  some']
    assert_equal(marked_lines, doctest_markup(marked_lines))
    marked_lines = ['any #2to3: here+1; line.replace("some", "boo") ',
                    ' some ']
    assert_equal(marked_lines, doctest_markup(marked_lines))
    marked_lines = ['>>> any #2to3: here+1; line.replace("some", "boo") ',
                    ' some ']
    exp_out = ['>>> any #2to3: here+1; line.replace("some", "boo") ',
               ' boo ']
    assert_equal(exp_out, doctest_markup(marked_lines))
    marked_lines = ['>>> any #2to3: next; line.replace("some", "boo") ',
                    ' some ']
    exp_out = ['>>> any #2to3: next; line.replace("some", "boo") ',
               ' boo ']
    assert_equal(exp_out, doctest_markup(marked_lines))
    assert_equal(['>>> woo #2to3: here ; line.replace("wow", "woo") '],
                 doctest_markup(
                     ['>>> wow #2to3: here ; line.replace("wow", "woo") ']))
    assert_equal(['>>> woo #2to3: here ; line.replace("wow", "woo") \n'],
                 doctest_markup(
                     ['>>> wow #2to3: here ; line.replace("wow", "woo") \n']))
    assert_equal(['>>> woo #2to3: here ; replace("wow", "woo") '],
                 doctest_markup(
                     ['>>> wow #2to3: here ; replace("wow", "woo") ']))
    assert_equal([">>> b'hello'  #2to3: here; bytes"],
                 doctest_markup(
                     [">>> 'hello'  #2to3: here; bytes"]))
    assert_equal(['>>> some #2to3: next; bytes\n', "    b'TRACK'\n"],
                 doctest_markup(
                     ['>>> some #2to3: next; bytes\n', "    'TRACK'\n"]))

def test_byter():
    # Test bytes formatter
    assert_equal('(b"hello \' world", b\'again\')',
                 byter('("hello \' world", "again")'))
    line = "_ = bio.write(' ' * 10)"
    assert_equal(
        byter(line),
        "_ = bio.write(b' ' * 10)")

