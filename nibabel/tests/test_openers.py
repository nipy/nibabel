# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Test for openers module """
import os
import contextlib
from gzip import GzipFile
from io import BytesIO, UnsupportedOperation
from distutils.version import StrictVersion

from numpy.compat.py3k import asstr, asbytes
from ..openers import Opener, ImageOpener, HAVE_INDEXED_GZIP, BZ2File
from ..tmpdirs import InTemporaryDirectory
from ..volumeutils import BinOpener

import unittest
from unittest import mock
import pytest
from ..testing import error_warnings


class Lunk(object):
    # bare file-like for testing
    closed = False

    def __init__(self, message):
        self.message = message

    def write(self):
        pass

    def read(self):
        return self.message


def test_Opener():
    # Test default mode is 'rb'
    fobj = Opener(__file__)
    assert fobj.mode == 'rb'
    fobj.close()
    # That it's a context manager
    with Opener(__file__) as fobj:
        assert fobj.mode == 'rb'
    # That we can set the mode
    with Opener(__file__, 'r') as fobj:
        assert fobj.mode == 'r'
    # with keyword arguments
    with Opener(__file__, mode='r') as fobj:
        assert fobj.mode == 'r'
    # fileobj returns fileobj passed through
    message = b"Wine?  Wouldn't you?"
    for obj in (BytesIO(message), Lunk(message)):
        with Opener(obj) as fobj:
            assert fobj.read() == message
        # Which does not close the object
        assert not obj.closed
        # mode is gently ignored
        fobj = Opener(obj, mode='r')


def test_Opener_various():
    # Check we can do all sorts of files here
    message = b"Oh what a giveaway"
    bz2_fileno = hasattr(BZ2File, 'fileno')
    if HAVE_INDEXED_GZIP:
        import indexed_gzip as igzip
    with InTemporaryDirectory():
        sobj = BytesIO()
        for input in ('test.txt',
                      'test.txt.gz',
                      'test.txt.bz2',
                      sobj):
            with Opener(input, 'wb') as fobj:
                fobj.write(message)
                assert fobj.tell() == len(message)
            if input == sobj:
                input.seek(0)
            with Opener(input, 'rb') as fobj:
                message_back = fobj.read()
                assert message == message_back
                if input == sobj:
                    # Fileno is unsupported for BytesIO
                    with pytest.raises(UnsupportedOperation):
                        fobj.fileno()
                elif input.endswith('.bz2') and not bz2_fileno:
                    with pytest.raises(AttributeError):
                        fobj.fileno()
                # indexed gzip is used by default, and drops file
                # handles by default, so we don't have a fileno.
                elif input.endswith('gz') and HAVE_INDEXED_GZIP and \
                     StrictVersion(igzip.__version__) >= StrictVersion('0.7.0'):
                    with pytest.raises(igzip.NoHandleError):
                        fobj.fileno()
                else:
                    # Just check there is a fileno
                    assert fobj.fileno() != 0


def test_BinOpener():
    with error_warnings():
        with pytest.raises(DeprecationWarning):
            BinOpener('test.txt', 'r')


class MockIndexedGzipFile(GzipFile):
    def __init__(self, *args, **kwargs):
        self._drop_handles = kwargs.pop('drop_handles', False)
        super(MockIndexedGzipFile, self).__init__(*args, **kwargs)


@contextlib.contextmanager
def patch_indexed_gzip(state):
    # Make it look like we do (state==True) or do not (state==False) have
    # the indexed gzip module.
    if state:
        values = (True, MockIndexedGzipFile)
    else:
        values = (False, GzipFile)
    with mock.patch('nibabel.openers.HAVE_INDEXED_GZIP', values[0]), \
         mock.patch('nibabel.openers.IndexedGzipFile', values[1],
                    create=True):
        yield


def test_Opener_gzip_type():
    # Test that BufferedGzipFile or IndexedGzipFile are used as appropriate

    data = 'this is some test data'
    fname = 'test.gz'

    with InTemporaryDirectory():

        # make some test data
        with GzipFile(fname, mode='wb') as f:
            f.write(data.encode())

        # Each test is specified by a tuple containing:
        #   (indexed_gzip present, Opener kwargs, expected file type)
        tests = [
            (False, {'mode' : 'rb', 'keep_open' : True},   GzipFile),
            (False, {'mode' : 'rb', 'keep_open' : False},  GzipFile),
            (False, {'mode' : 'rb', 'keep_open' : 'auto'}, GzipFile),
            (False, {'mode' : 'wb', 'keep_open' : True},   GzipFile),
            (False, {'mode' : 'wb', 'keep_open' : False},  GzipFile),
            (False, {'mode' : 'wb', 'keep_open' : 'auto'}, GzipFile),
            (True,  {'mode' : 'rb', 'keep_open' : True},   MockIndexedGzipFile),
            (True,  {'mode' : 'rb', 'keep_open' : False},  MockIndexedGzipFile),
            (True,  {'mode' : 'rb', 'keep_open' : 'auto'}, MockIndexedGzipFile),
            (True,  {'mode' : 'wb', 'keep_open' : True},   GzipFile),
            (True,  {'mode' : 'wb', 'keep_open' : False},  GzipFile),
            (True,  {'mode' : 'wb', 'keep_open' : 'auto'}, GzipFile),
        ]

        for test in tests:
            igzip_present, kwargs, expected = test
            with patch_indexed_gzip(igzip_present):
                assert isinstance(Opener(fname, **kwargs).fobj, expected)


class TestImageOpener(unittest.TestCase):
    def test_vanilla(self):
        # Test that ImageOpener does add '.mgz' as gzipped file type
        with InTemporaryDirectory():
            with ImageOpener('test.gz', 'w') as fobj:
                assert hasattr(fobj.fobj, 'compress')
            with ImageOpener('test.mgz', 'w') as fobj:
                assert hasattr(fobj.fobj, 'compress')

    @mock.patch.dict('nibabel.openers.ImageOpener.compress_ext_map')
    def test_new_association(self):
        def file_opener(fileish, mode):
            return open(fileish, mode)

        # Add the association
        n_associations = len(ImageOpener.compress_ext_map)
        ImageOpener.compress_ext_map['.foo'] = (file_opener, ('mode',))
        assert n_associations + 1 == len(ImageOpener.compress_ext_map)
        assert '.foo' in ImageOpener.compress_ext_map

        with InTemporaryDirectory():
            with ImageOpener('test.foo', 'w'):
                pass
            assert os.path.exists('test.foo')

        # Check this doesn't add anything to parent
        assert '.foo' not in Opener.compress_ext_map


def test_file_like_wrapper():
    # Test wrapper using BytesIO (full API)
    message = b"History of the nude in"
    sobj = BytesIO()
    fobj = Opener(sobj)
    assert fobj.tell() == 0
    fobj.write(message)
    assert fobj.tell() == len(message)
    fobj.seek(0)
    assert fobj.tell() == 0
    assert fobj.read(6) == message[:6]
    assert not fobj.closed
    fobj.close()
    assert fobj.closed
    # Added the fileobj name
    assert fobj.name is None


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
        for ext in ('gz', 'bz2', 'GZ', 'gZ', 'BZ2', 'Bz2'):
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
                assert sizes['default'] == sizes[default_val]
                assert sizes[1] > sizes[5]


def test_compressed_ext_case():
    # Test openers usually ignore case for compressed exts
    contents = b'palindrome of Bolton is notlob'

    class StrictOpener(Opener):
        compress_ext_icase = False
    exts = ('gz', 'bz2', 'GZ', 'gZ', 'BZ2', 'Bz2')
    with InTemporaryDirectory():
        # Make a basic file to check type later
        with open(__file__, 'rb') as a_file:
            file_class = type(a_file)
        for ext in exts:
            fname = 'test.' + ext
            with Opener(fname, 'wb') as fobj:
                fobj.write(contents)
            with Opener(fname, 'rb') as fobj:
                assert fobj.read() == contents
            os.unlink(fname)
            with StrictOpener(fname, 'wb') as fobj:
                fobj.write(contents)
            with StrictOpener(fname, 'rb') as fobj:
                assert fobj.read() == contents
            lext = ext.lower()
            if lext != ext:  # extension should not be recognized -> file
                assert isinstance(fobj.fobj, file_class)
            elif lext == 'gz':
                try:
                    from ..openers import IndexedGzipFile
                except ImportError:
                    IndexedGzipFile = GzipFile
                assert isinstance(fobj.fobj, (GzipFile, IndexedGzipFile))
            else:
                assert isinstance(fobj.fobj, BZ2File)


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
                assert fobj.name == exp_name


def test_set_extensions():
    # Test that we can add extensions that are compressed
    with InTemporaryDirectory():
        with Opener('test.gz', 'w') as fobj:
            assert hasattr(fobj.fobj, 'compress')
        with Opener('test.glrph', 'w') as fobj:
            assert not hasattr(fobj.fobj, 'compress')

        class MyOpener(Opener):
            compress_ext_map = Opener.compress_ext_map.copy()
            compress_ext_map['.glrph'] = Opener.gz_def
        with MyOpener('test.glrph', 'w') as fobj:
            assert hasattr(fobj.fobj, 'compress')


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
                assert not fobj.closed
            fobj.close_if_mine()
            is_str = type(input) is type('')
            if has_closed:
                assert fobj.closed == is_str


def test_iter():
    # Check we can iterate over lines, if the underlying file object allows it
    lines = \
        """On the
blue ridged mountains
of
virginia
""".split('\n')
    with InTemporaryDirectory():
        sobj = BytesIO()
        for input, does_t in (('test.txt', True),
                              ('test.txt.gz', False),
                              ('test.txt.bz2', False),
                              (sobj, True)):
            with Opener(input, 'wb') as fobj:
                for line in lines:
                    fobj.write(asbytes(line + os.linesep))
            with Opener(input, 'rb') as fobj:
                for back_line, line in zip(fobj, lines):
                    assert asstr(back_line).rstrip() == line
            if not does_t:
                continue
            with Opener(input, 'rt') as fobj:
                for back_line, line in zip(fobj, lines):
                    assert back_line.rstrip() == line
        lobj = Opener(Lunk(''))
        with pytest.raises(TypeError):
            list(lobj)
