import os
import unittest
import numpy as np
from os.path import join as pjoin

from io import BytesIO
from nibabel.py3k import asbytes

from ..array_sequence import ArraySequence
from ..tractogram import Tractogram
from ..tractogram_file import HeaderWarning, HeaderError
from ..tractogram_file import DataError

from .. import tck as tck_module
from ..tck import TckFile

from nose.tools import assert_equal, assert_raises, assert_true
from numpy.testing import assert_array_equal
from nibabel.testing import data_path, clear_and_catch_warnings
from .test_tractogram import assert_tractogram_equal

DATA = {}


def setup():
    global DATA

    DATA['empty_tck_fname'] = pjoin(data_path, "empty.tck")
    # simple.tck contains only streamlines
    DATA['simple_tck_fname'] = pjoin(data_path, "simple.tck")
    DATA['simple_tck_big_endian_fname'] = pjoin(data_path,
                                                "simple_big_endian.tck")
    # standard.tck contains only streamlines
    DATA['standard_tck_fname'] = pjoin(data_path, "standard.tck")
    DATA['matlab_nan_tck_fname'] = pjoin(data_path, "matlab_nan.tck")

    DATA['streamlines'] = [np.arange(1 * 3, dtype="f4").reshape((1, 3)),
                           np.arange(2 * 3, dtype="f4").reshape((2, 3)),
                           np.arange(5 * 3, dtype="f4").reshape((5, 3))]

    DATA['empty_tractogram'] = Tractogram(affine_to_rasmm=np.eye(4))
    DATA['simple_tractogram'] = Tractogram(DATA['streamlines'],
                                           affine_to_rasmm=np.eye(4))


class TestTCK(unittest.TestCase):

    def test_load_empty_file(self):
        for lazy_load in [False, True]:
            tck = TckFile.load(DATA['empty_tck_fname'], lazy_load=lazy_load)
            assert_tractogram_equal(tck.tractogram, DATA['empty_tractogram'])

    def test_load_simple_file(self):
        for lazy_load in [False, True]:
            tck = TckFile.load(DATA['simple_tck_fname'], lazy_load=lazy_load)
            assert_tractogram_equal(tck.tractogram, DATA['simple_tractogram'])

        # Force TCK loading to use buffering.
        buffer_size = 1. / 1024**2  # 1 bytes
        hdr = TckFile._read_header(DATA['simple_tck_fname'])
        tck_reader = TckFile._read(DATA['simple_tck_fname'], hdr, buffer_size)
        streamlines = ArraySequence(tck_reader)
        tractogram = Tractogram(streamlines)
        tractogram.affine_to_rasmm = np.eye(4)
        tck = TckFile(tractogram, header=hdr)
        assert_tractogram_equal(tck.tractogram, DATA['simple_tractogram'])

    def test_load_matlab_nan_file(self):
        for lazy_load in [False, True]:
            tck = TckFile.load(DATA['matlab_nan_tck_fname'], lazy_load=lazy_load)
            streamlines = list(tck.tractogram.streamlines)
            assert_equal(len(streamlines), 1)
            assert_equal(streamlines[0].shape, (108, 3))

    def test_writeable_data(self):
        data = DATA['simple_tractogram']
        for key in ('simple_tck_fname', 'simple_tck_big_endian_fname'):
            for lazy_load in [False, True]:
                tck = TckFile.load(DATA[key], lazy_load=lazy_load)
                for actual, expected_tgi in zip(tck.streamlines, data):
                    assert_array_equal(actual, expected_tgi.streamline)
                    # Test we can write to arrays
                    assert_true(actual.flags.writeable)
                    actual[0, 0] = 99

    def test_load_simple_file_in_big_endian(self):
        for lazy_load in [False, True]:
            tck = TckFile.load(DATA['simple_tck_big_endian_fname'],
                               lazy_load=lazy_load)
            assert_tractogram_equal(tck.tractogram, DATA['simple_tractogram'])
            assert_equal(tck.header['datatype'], 'Float32BE')

    def test_load_file_with_wrong_information(self):
        tck_file = open(DATA['simple_tck_fname'], 'rb').read()

        # Simulate a TCK file where `datatype` has not the right endianness.
        new_tck_file = tck_file.replace(asbytes("Float32LE"),
                                        asbytes("Float32BE"))
        assert_raises(DataError, TckFile.load, BytesIO(new_tck_file))

        # Simulate a TCK file with unsupported `datatype`.
        new_tck_file = tck_file.replace(asbytes("Float32LE"),
                                        asbytes("int32"))
        assert_raises(HeaderError, TckFile.load, BytesIO(new_tck_file))

        # Simulate a TCK file with no `datatype` field.
        new_tck_file = tck_file.replace(b"datatype: Float32LE\n", b"")
        # Need to adjust data offset.
        new_tck_file = new_tck_file.replace(b"file: . 67\n", b"file: . 47\n")
        with clear_and_catch_warnings(record=True, modules=[tck_module]) as w:
            tck = TckFile.load(BytesIO(new_tck_file))
            assert_equal(len(w), 1)
            assert_true(issubclass(w[0].category, HeaderWarning))
            assert_true("Missing 'datatype'" in str(w[0].message))
            assert_array_equal(tck.header['datatype'], "Float32LE")

        # Simulate a TCK file with no `file` field.
        new_tck_file = tck_file.replace(b"\nfile: . 67", b"")
        with clear_and_catch_warnings(record=True, modules=[tck_module]) as w:
            tck = TckFile.load(BytesIO(new_tck_file))
            assert_equal(len(w), 1)
            assert_true(issubclass(w[0].category, HeaderWarning))
            assert_true("Missing 'file'" in str(w[0].message))
            assert_array_equal(tck.header['file'], ". 56")

        # Simulate a TCK file with `file` field pointing to another file.
        new_tck_file = tck_file.replace(b"file: . 67\n",
                                        b"file: dummy.mat 75\n")
        assert_raises(HeaderError, TckFile.load, BytesIO(new_tck_file))

        # Simulate a TCK file which is missing a streamline delimiter.
        eos = TckFile.FIBER_DELIMITER.tostring()
        eof = TckFile.EOF_DELIMITER.tostring()
        new_tck_file = tck_file[:-(len(eos) + len(eof))] + tck_file[-len(eof):]

        # Force TCK loading to use buffering.
        buffer_size = 1. / 1024**2  # 1 bytes
        hdr = TckFile._read_header(BytesIO(new_tck_file))
        tck_reader = TckFile._read(BytesIO(new_tck_file), hdr, buffer_size)
        assert_raises(DataError, list, tck_reader)

        # Simulate a TCK file which is missing the end-of-file delimiter.
        new_tck_file = tck_file[:-len(eof)]
        assert_raises(DataError, TckFile.load, BytesIO(new_tck_file))

    def test_write_empty_file(self):
        tractogram = Tractogram(affine_to_rasmm=np.eye(4))

        tck_file = BytesIO()
        tck = TckFile(tractogram)
        tck.save(tck_file)
        tck_file.seek(0, os.SEEK_SET)

        new_tck = TckFile.load(tck_file)
        assert_tractogram_equal(new_tck.tractogram, tractogram)

        new_tck_orig = TckFile.load(DATA['empty_tck_fname'])
        assert_tractogram_equal(new_tck.tractogram, new_tck_orig.tractogram)

        tck_file.seek(0, os.SEEK_SET)
        assert_equal(tck_file.read(),
                     open(DATA['empty_tck_fname'], 'rb').read())

    def test_write_simple_file(self):
        tractogram = Tractogram(DATA['streamlines'],
                                affine_to_rasmm=np.eye(4))

        tck_file = BytesIO()
        tck = TckFile(tractogram)
        tck.save(tck_file)
        tck_file.seek(0, os.SEEK_SET)

        new_tck = TckFile.load(tck_file)
        assert_tractogram_equal(new_tck.tractogram, tractogram)

        new_tck_orig = TckFile.load(DATA['simple_tck_fname'])
        assert_tractogram_equal(new_tck.tractogram, new_tck_orig.tractogram)

        tck_file.seek(0, os.SEEK_SET)
        assert_equal(tck_file.read(),
                     open(DATA['simple_tck_fname'], 'rb').read())

        # TCK file containing not well formatted entries in its header.
        tck_file = BytesIO()
        tck = TckFile(tractogram)
        tck.header['new_entry'] = 'value\n'  # \n not allowed
        assert_raises(HeaderError, tck.save, tck_file)

        tck.header['new_entry'] = 'val:ue'  # : not allowed
        assert_raises(HeaderError, tck.save, tck_file)

    def test_load_write_file(self):
        for fname in [DATA['empty_tck_fname'],
                      DATA['simple_tck_fname']]:
            for lazy_load in [False, True]:
                tck = TckFile.load(fname, lazy_load=lazy_load)
                tck_file = BytesIO()
                tck.save(tck_file)

                loaded_tck = TckFile.load(fname, lazy_load=False)
                assert_tractogram_equal(loaded_tck.tractogram, tck.tractogram)

                # Check that the written file is the same as the one read.
                tck_file.seek(0, os.SEEK_SET)
                assert_equal(tck_file.read(), open(fname, 'rb').read())

        # Save tractogram that has an affine_to_rasmm.
        for lazy_load in [False, True]:
            tck = TckFile.load(DATA['simple_tck_fname'], lazy_load=lazy_load)
            affine = np.eye(4)
            affine[0, 0] *= -1  # Flip in X
            tractogram = Tractogram(tck.streamlines, affine_to_rasmm=affine)

            new_tck = TckFile(tractogram, tck.header)
            tck_file = BytesIO()
            new_tck.save(tck_file)
            tck_file.seek(0, os.SEEK_SET)

            loaded_tck = TckFile.load(tck_file, lazy_load=False)
            assert_tractogram_equal(loaded_tck.tractogram,
                                    tractogram.to_world(lazy=True))

    def test_str(self):
        tck = TckFile.load(DATA['simple_tck_fname'])
        str(tck)  # Simply test it's not failing when called.
