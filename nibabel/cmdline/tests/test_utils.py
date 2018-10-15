# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test scripts

Test running scripts
"""

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)

from nibabel.cmdline.utils import *


def test_table2string():
    assert_equal(table2string([["A", "B", "C", "D"], ["E", "F", "G", "H"]]), "A B C D\nE F G H\n")
    assert_equal(table2string([["Let's", "Make", "Tests", "And"], ["Have", "Lots", "Of", "Fun"],
                               ["With", "Python", "Guys", "!"]]), "Let's  Make  Tests And\n Have  Lots    Of  Fun"+
                                                                  "\n With Python  Guys  !\n")


def test_ap():
    assert_equal(ap([1, 2], "%2d"), " 1,  2")
    assert_equal(ap([1, 2], "%3d"), "  1,   2")
    assert_equal(ap([1, 2], "%-2d"), "1 , 2 ")
    assert_equal(ap([1, 2], "%d", "+"), "1+2")
    assert_equal(ap([1, 2, 3], "%d", "-"), "1-2-3")


def test_safe_get():
    class TestObject:
        def __init__(self, test=None):
            self.test = test

        def get_test(self):
            return self.test

    test = TestObject()
    test.test = 2

    assert_equal(safe_get(test, "test"), 2)
    assert_equal(safe_get(test, "failtest"), "-")
