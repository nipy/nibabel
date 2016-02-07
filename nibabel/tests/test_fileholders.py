""" Testing fileholders
"""

from ..externals.six import BytesIO


from ..fileholders import FileHolder

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import assert_equal
from nose.tools import assert_false
from nose.tools import assert_true


def test_init():
    fh = FileHolder('a_fname')
    assert_equal(fh.filename, 'a_fname')
    assert_true(fh.fileobj is None)
    assert_equal(fh.pos, 0)
    sio0 = BytesIO()
    fh = FileHolder('a_test', sio0)
    assert_equal(fh.filename, 'a_test')
    assert_true(fh.fileobj is sio0)
    assert_equal(fh.pos, 0)
    fh = FileHolder('a_test_2', sio0, 3)
    assert_equal(fh.filename, 'a_test_2')
    assert_true(fh.fileobj is sio0)
    assert_equal(fh.pos, 3)


def test_same_file_as():
    fh = FileHolder('a_fname')
    assert_true(fh.same_file_as(fh))
    fh2 = FileHolder('a_test')
    assert_false(fh.same_file_as(fh2))
    sio0 = BytesIO()
    fh3 = FileHolder('a_fname', sio0)
    fh4 = FileHolder('a_fname', sio0)
    assert_true(fh3.same_file_as(fh4))
    assert_false(fh3.same_file_as(fh))
    fh5 = FileHolder(fileobj=sio0)
    fh6 = FileHolder(fileobj=sio0)
    assert_true(fh5.same_file_as(fh6))
    # Not if the filename is the same
    assert_false(fh5.same_file_as(fh3))
    # pos doesn't matter
    fh4_again = FileHolder('a_fname', sio0, pos=4)
    assert_true(fh3.same_file_as(fh4_again))


def test_file_like():
    # Test returning file object or filename
    fh = FileHolder('a_fname')
    assert_equal(fh.file_like, 'a_fname')
    bio = BytesIO()
    fh = FileHolder(fileobj=bio)
    assert_true(fh.file_like is bio)
    fh = FileHolder('a_fname', fileobj=bio)
    assert_true(fh.file_like is bio)
