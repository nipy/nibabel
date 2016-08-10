# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import warnings

from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_raises)
from nibabel.testing import clear_and_catch_warnings
from nibabel.tmpdirs import InTemporaryDirectory


from .test_parse_gifti_fast import (DATA_FILE1, DATA_FILE2, DATA_FILE3,
                                    DATA_FILE4, DATA_FILE5, DATA_FILE6)


class TestGiftiIO(object):

    def setUp(self):
        with clear_and_catch_warnings() as w:
            warnings.simplefilter('always', DeprecationWarning)
            assert_equal(len(w), 1)


def test_read_deprecated():
    with clear_and_catch_warnings() as w:
        warnings.simplefilter('always', DeprecationWarning)
        from nibabel.gifti.giftiio import read, write

        img = read(DATA_FILE1)
        assert_equal(len(w), 1)
        with InTemporaryDirectory():
            write(img, 'test.gii')
        assert_equal(len(w), 2)
