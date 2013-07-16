""" Testing Siemens CSA header reader
"""
from os.path import join as pjoin
import gzip

import numpy as np

from .. import csareader as csa
from .. import dwiparams as dwp

from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)

from numpy.testing import assert_array_equal, assert_array_almost_equal

from .test_dicomwrappers import (have_dicom, dicom_test,
                                 IO_DATA_PATH, DATA, DATA_FILE)

CSA2_B0 = open(pjoin(IO_DATA_PATH, 'csa2_b0.bin'), 'rb').read()
CSA2_B1000 = open(pjoin(IO_DATA_PATH, 'csa2_b1000.bin'), 'rb').read()
CSA2_0len = gzip.open(pjoin(IO_DATA_PATH, 'csa2_zero_len.bin.gz'), 'rb').read()


@dicom_test
def test_csa_header_read():
    hdr = csa.get_csa_header(DATA, 'image')
    assert_equal(hdr['n_tags'],83)
    assert_equal(csa.get_csa_header(DATA,'series')['n_tags'],65)
    assert_raises(ValueError, csa.get_csa_header, DATA,'xxxx')
    assert_true(csa.is_mosaic(hdr))
    # Get a shallow copy of the data, lacking the CSA marker
    # Need to do it this way because del appears broken in pydicom 0.9.7
    from dicom.dataset import Dataset
    data2 = Dataset()
    for element in DATA:
        if (element.tag.group, element.tag.elem) != (0x29, 0x10):
            data2.add(element)
    assert_equal(csa.get_csa_header(data2, 'image'), None)
    # Add back the marker - CSA works again
    data2[(0x29, 0x10)] = DATA[(0x29, 0x10)]
    assert_true(csa.is_mosaic(csa.get_csa_header(data2, 'image')))


def test_csas0():
    for csa_str in (CSA2_B0, CSA2_B1000):
        csa_info = csa.read(csa_str)
        assert_equal(csa_info['type'], 2)
        assert_equal(csa_info['n_tags'], 83)
        tags = csa_info['tags']
        assert_equal(len(tags), 83)
        n_o_m = tags['NumberOfImagesInMosaic']
        assert_equal(n_o_m['items'], [48])
    csa_info = csa.read(CSA2_B1000)
    b_matrix = csa_info['tags']['B_matrix']
    assert_equal(len(b_matrix['items']), 6)
    b_value = csa_info['tags']['B_value']
    assert_equal(b_value['items'], [1000])


def test_csa_len0():
    # We did get a failure for item with item_len of 0 - gh issue #92
    csa_info = csa.read(CSA2_0len)
    assert_equal(csa_info['type'], 2)
    assert_equal(csa_info['n_tags'], 44)
    tags = csa_info['tags']
    assert_equal(len(tags), 44)


def test_csa_params():
    for csa_str in (CSA2_B0, CSA2_B1000):
        csa_info = csa.read(csa_str)
        n_o_m = csa.get_n_mosaic(csa_info)
        assert_equal(n_o_m, 48)
        snv = csa.get_slice_normal(csa_info)
        assert_equal(snv.shape, (3,))
        assert_true(np.allclose(1,
                np.sqrt((snv * snv).sum())))
        amt = csa.get_acq_mat_txt(csa_info)
        assert_equal(amt, '128p*128')
    csa_info = csa.read(CSA2_B0)
    b_matrix = csa.get_b_matrix(csa_info)
    assert_equal(b_matrix, None)
    b_value = csa.get_b_value(csa_info)
    assert_equal(b_value, 0)
    g_vector = csa.get_g_vector(csa_info)
    assert_equal(g_vector, None)
    csa_info = csa.read(CSA2_B1000)
    b_matrix = csa.get_b_matrix(csa_info)
    assert_equal(b_matrix.shape, (3,3))
    # check (by absence of error) that the B matrix is positive
    # semi-definite.
    q = dwp.B2q(b_matrix)
    b_value = csa.get_b_value(csa_info)
    assert_equal(b_value, 1000)
    g_vector = csa.get_g_vector(csa_info)
    assert_equal(g_vector.shape, (3,))
    assert_true(
        np.allclose(1, np.sqrt((g_vector * g_vector).sum())))


def test_ice_dims():
    ex_dims0 = ['X', '1', '1', '1', '1', '1', '1',
                '48', '1', '1', '1', '1', '201']
    ex_dims1 = ['X', '1', '1', '1', '2', '1', '1',
               '48', '1', '1', '1', '1', '201']
    for csa_str, ex_dims in ((CSA2_B0, ex_dims0),
                             (CSA2_B1000, ex_dims1)):
        csa_info = csa.read(csa_str)
        assert_equal(csa.get_ice_dims(csa_info),
                           ex_dims)
    assert_equal(csa.get_ice_dims({}), None)
