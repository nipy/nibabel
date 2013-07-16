# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Test for openers module '''
from ..externals.six import BytesIO

from ..tmpdirs import InTemporaryDirectory

from ..openers import Opener

from nose.tools import assert_true, assert_false, assert_equal, assert_raises

class Lunk(object):
    # bare file-like for testing
    closed = False
    def __init__(self, message):
        self.message=message
    def write(self):
        pass
    def read(self):
        return self.message


def test_Opener():
    # Test default mode is 'rb'
    fobj = Opener(__file__)
    assert_equal(fobj.mode, 'rb')
    fobj.close()
    # That it's a context manager
    with Opener(__file__) as fobj:
        assert_equal(fobj.mode, 'rb')
    # That we can set the mode
    with Opener(__file__, 'r') as fobj:
        assert_equal(fobj.mode, 'r')
    # with keyword arguments
    with Opener(__file__, mode='r') as fobj:
        assert_equal(fobj.mode, 'r')
    # fileobj returns fileobj passed through
    message = b"Wine?  Wouldn't you?"
    for obj in (BytesIO(message), Lunk(message)):
        with Opener(obj) as fobj:
            assert_equal(fobj.read(), message)
        # Which does not close the object
        assert_false(obj.closed)
        # mode is gently ignored
        fobj = Opener(obj, mode='r')


def test_Opener_various():
    # Check we can do all sorts of files here
    message = b"Oh what a giveaway"
    with InTemporaryDirectory():
        sobj = BytesIO()
        for input in ('test.txt',
                      'test.txt.gz',
                      'test.txt.bz2',
                      sobj):
            with Opener(input, 'wb') as fobj:
                fobj.write(message)
                assert_equal(fobj.tell(), len(message))
            if input == sobj:
                input.seek(0)
            with Opener(input, 'rb') as fobj:
                message_back = fobj.read()
                assert_equal(message, message_back)


def test_file_like_wrapper():
    # Test wrapper using BytesIO (full API)
    message = b"History of the nude in"
    sobj = BytesIO()
    fobj = Opener(sobj)
    assert_equal(fobj.tell(), 0)
    fobj.write(message)
    assert_equal(fobj.tell(), len(message))
    fobj.seek(0)
    assert_equal(fobj.tell(), 0)
    assert_equal(fobj.read(6), message[:6])
    assert_false(fobj.closed)
    fobj.close()
    assert_true(fobj.closed)
    # Added the fileobj name
    assert_equal(fobj.name, None)


def test_compressionlevel():
    # Check default and set compression level
    with open(__file__, 'rb') as fobj:
        my_self = fobj.read()
    # bzip2 needs a fairly large file to show differences in compression level
    many_selves = my_self * 50
    # Test we can set default compression at class level
    class MyOpener(Opener):
        default_compresslevel = 5
    with InTemporaryDirectory():
        for ext in ('gz', 'bz2'):
            for opener, default_val in ((Opener, 1), (MyOpener, 5)):
                sizes = {}
                for compresslevel in ('default', 1, 5):
                    fname = 'test.' + ext
                    kwargs = {'mode': 'wb'}
                    if compresslevel != 'default':
                        kwargs['compresslevel'] = compresslevel
                    with opener(fname, **kwargs) as fobj:
                        fobj.write(many_selves)
                    with open(fname, 'rb') as fobj:
                        my_selves_smaller = fobj.read()
                    sizes[compresslevel] = len(my_selves_smaller)
                assert_equal(sizes['default'], sizes[default_val])
                assert_true(sizes[1] > sizes[5])


def test_name():
    # The wrapper gives everything a name, maybe None
    sobj = BytesIO()
    lunk = Lunk('in ART')
    with InTemporaryDirectory():
        for input in ('test.txt',
                      'test.txt.gz',
                      'test.txt.bz2',
                      sobj,
                      lunk):
            exp_name = input if type(input) == type('') else None
            with Opener(input, 'wb') as fobj:
                assert_equal(fobj.name, exp_name)


def test_set_extensions():
    # Test that we can add extensions that are compressed
    with InTemporaryDirectory():
        with Opener('test.gz', 'w') as fobj:
            assert_true(hasattr(fobj.fobj, 'compress'))
        with Opener('test.glrph', 'w') as fobj:
            assert_false(hasattr(fobj.fobj, 'compress'))
        class MyOpener(Opener):
            compress_ext_map = Opener.compress_ext_map.copy()
            compress_ext_map['.glrph'] = Opener.gz_def
        with MyOpener('test.glrph', 'w') as fobj:
            assert_true(hasattr(fobj.fobj, 'compress'))


def test_close_if_mine():
    # Test that we close the file iff we opened it
    with InTemporaryDirectory():
        sobj = BytesIO()
        lunk = Lunk('')
        for input in ('test.txt',
                      'test.txt.gz',
                      'test.txt.bz2',
                      sobj,
                      lunk):
            fobj = Opener(input, 'wb')
            # gzip objects have no 'closed' attribute
            has_closed = hasattr(fobj.fobj, 'closed')
            if has_closed:
                assert_false(fobj.closed)
            fobj.close_if_mine()
            is_str = type(input) is type('')
            if has_closed:
                assert_equal(fobj.closed, is_str)
