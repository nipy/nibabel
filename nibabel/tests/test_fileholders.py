""" Testing fileholders
"""

from ..py3k import BytesIO

from ..fileholders import FileHolder, FileHolderError, copy_file_map
from ..tmpdirs import InTemporaryDirectory
from ..stampers import Stamper

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false,
                        assert_equal, assert_not_equal,
                        assert_raises)


def test_init():
    fh = FileHolder('a_fname')
    assert_equal(fh.filename, 'a_fname')
    assert_true(fh.fileobj is None)
    assert_equal(fh.pos, 0)
    bio = BytesIO()
    fh = FileHolder('a_test', bio)
    assert_equal(fh.filename, 'a_test')
    assert_true(fh.fileobj is bio)
    assert_equal(fh.pos, 0)
    fh = FileHolder('a_test_2', bio, 3)
    assert_equal(fh.filename, 'a_test_2')
    assert_true(fh.fileobj is bio)
    assert_equal(fh.pos, 3)


def test_same_file_as():
    fh = FileHolder('a_fname')
    assert_true(fh.same_file_as(fh))
    fh2 = FileHolder('a_test')
    assert_false(fh.same_file_as(fh2))
    bio = BytesIO()
    fh3 = FileHolder('a_fname', bio)
    fh4 = FileHolder('a_fname', bio)
    assert_true(fh3.same_file_as(fh4))
    assert_false(fh3.same_file_as(fh))
    fh5 = FileHolder(fileobj=bio)
    fh6 = FileHolder(fileobj=bio)
    assert_true(fh5.same_file_as(fh6))
    # Not if the filename is the same
    assert_false(fh5.same_file_as(fh3))
    # pos doesn't matter
    fh4_again = FileHolder('a_fname', bio, pos=4)
    assert_true(fh3.same_file_as(fh4_again))


def test_stamping():
    # Test stamping works as expected
    stamper = Stamper()
    fh1 = FileHolder('a_fname')
    fh2 = FileHolder('a_fname')
    assert_equal(stamper(fh1), stamper(fh2))
    fh3 = FileHolder('a_test')
    assert_not_equal(stamper(fh1), stamper(fh3))
    bio = BytesIO()
    fh4 = FileHolder('a_fname', bio)
    fh5 = FileHolder('a_fname', bio)
    assert_equal(stamper(fh4), stamper(fh5))
    fh6 = FileHolder('a_fname2', bio)
    assert_not_equal(stamper(fh4), stamper(fh6))
    assert_equal((fh4.pos, fh5.pos), (0, 0))
    fh5.pos = 1
    assert_not_equal(stamper(fh4), stamper(fh5))
    fh4 = FileHolder(fileobj=bio)
    fh5 = FileHolder(fileobj=bio)
    assert_equal(stamper(fh4), stamper(fh5))
    assert_equal((fh4.pos, fh5.pos), (0, 0))
    fh5.pos = 1
    assert_not_equal(stamper(fh4), stamper(fh5))


def test_copy_file_map():
    # Test copy of fileholder using stamping
    bio = BytesIO()
    fm = dict(one=FileHolder('a_fname', bio), two=FileHolder('a_fname2'))
    fm2 = copy_file_map(fm)
    stamper = Stamper()
    assert_equal(stamper(fm), stamper(fm2))
    # Check you can modify the copies independently
    fm['one'].pos = 2
    assert_not_equal(stamper(fm), stamper(fm2))
