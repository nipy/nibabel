""" Testing dft
"""

import os
from os.path import join as pjoin, dirname
from ..externals.six import BytesIO
from ..testing import suppress_warnings

import numpy as np

with suppress_warnings():
    from .. import dft
from .. import nifti1

from nose import SkipTest
from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)

# Shield optional package imports
from ..optpkg import optional_package

# setup_module will raise SkipTest if no dicom to import
from nibabel.pydicom_compat import have_dicom

PImage, have_pil, _ = optional_package('PIL.Image')
pil_test = np.testing.dec.skipif(not have_pil, 'could not import PIL.Image')

data_dir = pjoin(dirname(__file__), 'data')


def setup_module():
    if os.name == 'nt':
        raise SkipTest('FUSE not available for windows, skipping dft tests')
    if not have_dicom:
        raise SkipTest('Need pydicom for dft tests, skipping')


def test_init():
    dft.clear_cache()
    dft.update_cache(data_dir)


def test_study():
    studies = dft.get_studies(data_dir)
    assert_equal(len(studies), 1)
    assert_equal(studies[0].uid,
                 '1.3.12.2.1107.5.2.32.35119.30000010011408520750000000022')
    assert_equal(studies[0].date, '20100114')
    assert_equal(studies[0].time, '121314.000000')
    assert_equal(studies[0].comments, 'dft study comments')
    assert_equal(studies[0].patient_name, 'dft patient name')
    assert_equal(studies[0].patient_id, '1234')
    assert_equal(studies[0].patient_birth_date, '19800102')
    assert_equal(studies[0].patient_sex, 'F')


def test_series():
    studies = dft.get_studies(data_dir)
    assert_equal(len(studies[0].series), 1)
    ser = studies[0].series[0]
    assert_equal(ser.uid,
                 '1.3.12.2.1107.5.2.32.35119.2010011420292594820699190.0.0.0')
    assert_equal(ser.number, '12')
    assert_equal(ser.description, 'CBU_DTI_64D_1A')
    assert_equal(ser.rows, 256)
    assert_equal(ser.columns, 256)
    assert_equal(ser.bits_allocated, 16)
    assert_equal(ser.bits_stored, 12)


def test_storage_instances():
    studies = dft.get_studies(data_dir)
    sis = studies[0].series[0].storage_instances
    assert_equal(len(sis), 2)
    assert_equal(sis[0].instance_number, 1)
    assert_equal(sis[1].instance_number, 2)
    assert_equal(sis[0].uid,
                 '1.3.12.2.1107.5.2.32.35119.2010011420300180088599504.0')
    assert_equal(sis[1].uid,
                 '1.3.12.2.1107.5.2.32.35119.2010011420300180088599504.1')


def test_storage_instance():
    pass


@pil_test
def test_png():
    studies = dft.get_studies(data_dir)
    data = studies[0].series[0].as_png()
    im = PImage.open(BytesIO(data))
    assert_equal(im.size, (256, 256))


def test_nifti():
    studies = dft.get_studies(data_dir)
    data = studies[0].series[0].as_nifti()
    assert_equal(len(data), 352 + 2 * 256 * 256 * 2)
    h = nifti1.Nifti1Header(data[:348])
    assert_equal(h.get_data_shape(), (256, 256, 2))
