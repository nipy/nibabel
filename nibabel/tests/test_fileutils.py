# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
# ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Testing fileutils module
"""


from ..fileutils import read_zt_byte_strings

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


from ..tmpdirs import InTemporaryDirectory


def test_read_zt_byte_strings():
    # sample binary block
    binary = b'test.fmr\x00test.prt\x00something'
    with InTemporaryDirectory():
        # create a tempfile
        path = 'test.bin'
        fwrite = open(path, 'wb')
        # write the binary block to it
        fwrite.write(binary)
        fwrite.close()
        # open it again
        fread = open(path, 'rb')
        # test readout of one string
        assert_equal(read_zt_byte_strings(fread), [b'test.fmr'])
        # test new file position
        assert_equal(fread.tell(), 9)
        # manually rewind
        fread.seek(0)
        # test readout of two strings
        assert_equal(read_zt_byte_strings(fread, 2),
                     [b'test.fmr', b'test.prt'])
        assert_equal(fread.tell(), 18)
        # test readout of more strings than present
        fread.seek(0)
        assert_raises(ValueError, read_zt_byte_strings, fread, 3)
        fread.seek(9)
        assert_raises(ValueError, read_zt_byte_strings, fread, 2)
        # Try with a small bufsize
        fread.seek(0)
        assert_equal(read_zt_byte_strings(fread, 2, 4),
                     [b'test.fmr', b'test.prt'])
        fread.close()
