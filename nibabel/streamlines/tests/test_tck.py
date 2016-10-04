import os
import unittest
import numpy as np
from os.path import join as pjoin

from nibabel.externals.six import BytesIO
from nibabel.py3k import asbytes

from nose.tools import assert_equal, assert_raises

from nibabel.testing import data_path
from .test_tractogram import assert_tractogram_equal
from ..array_sequence import ArraySequence
from ..tractogram import Tractogram
from ..tractogram_file import DataError

from ..tck import TckFile


DATA = {}


def setup():
    global DATA

    DATA['empty_tck_fname'] = pjoin(data_path, "empty.tck")
    # simple.trk contains only streamlines
    DATA['simple_tck_fname'] = pjoin(data_path, "simple.tck")
    DATA['simple_tck_big_endian_fname'] = pjoin(data_path,
                                                "simple_big_endian.tck")
    # standard.trk contains only streamlines
    DATA['standard_tck_fname'] = pjoin(data_path, "standard.tck")
    # complex.trk contains streamlines, scalars and properties
    # DATA['complex_tck_fname'] = pjoin(data_path, "complex.tck")

    DATA['streamlines'] = [np.arange(1*3, dtype="f4").reshape((1, 3)),
                           np.arange(2*3, dtype="f4").reshape((2, 3)),
                           np.arange(5*3, dtype="f4").reshape((5, 3))]

    # DATA['fa'] = [np.array([[0.2]], dtype="f4"),
    #               np.array([[0.3],
    #                         [0.4]], dtype="f4"),
    #               np.array([[0.5],
    #                         [0.6],
    #                         [0.6],
    #                         [0.7],
    #                         [0.8]], dtype="f4")]

    # DATA['colors'] = [np.array([(1, 0, 0)]*1, dtype="f4"),
    #                   np.array([(0, 1, 0)]*2, dtype="f4"),
    #                   np.array([(0, 0, 1)]*5, dtype="f4")]

    # DATA['mean_curvature'] = [np.array([1.11], dtype="f4"),
    #                           np.array([2.11], dtype="f4"),
    #                           np.array([3.11], dtype="f4")]

    # DATA['mean_torsion'] = [np.array([1.22], dtype="f4"),
    #                         np.array([2.22], dtype="f4"),
    #                         np.array([3.22], dtype="f4")]

    # DATA['mean_colors'] = [np.array([1, 0, 0], dtype="f4"),
    #                        np.array([0, 1, 0], dtype="f4"),
    #                        np.array([0, 0, 1], dtype="f4")]

    # DATA['data_per_point'] = {'colors': DATA['colors'],
    #                           'fa': DATA['fa']}
    # DATA['data_per_streamline'] = {'mean_curvature': DATA['mean_curvature'],
    #                                'mean_torsion': DATA['mean_torsion'],
    #                                'mean_colors': DATA['mean_colors']}

    DATA['empty_tractogram'] = Tractogram(affine_to_rasmm=np.eye(4))
    DATA['simple_tractogram'] = Tractogram(DATA['streamlines'],
                                           affine_to_rasmm=np.eye(4))
    # DATA['complex_tractogram'] = Tractogram(DATA['streamlines'],
    #                                         DATA['data_per_streamline'],
    #                                         DATA['data_per_point'],
    #                                         affine_to_rasmm=np.eye(4))


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

    def test_load_simple_file_in_big_endian(self):
        for lazy_load in [False, True]:
            tck = TckFile.load(DATA['simple_tck_big_endian_fname'],
                               lazy_load=lazy_load)
            assert_tractogram_equal(tck.tractogram, DATA['simple_tractogram'])
            assert_equal(tck.header['datatype'], 'Float32BE')

    def test_load_file_with_wrong_information(self):
        tck_file = open(DATA['simple_tck_fname'], 'rb').read()

        # Simulate a TCK file where `datatype` was incorrectly provided.
        new_tck_file = tck_file.replace(asbytes("Float32LE"),
                                        asbytes("Float32BE"))
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
