""" Testing dft
"""

import os
from os.path import join as pjoin, dirname
from io import BytesIO
from ..testing_pytest import suppress_warnings

with suppress_warnings():
    from .. import dft
from .. import nifti1

import unittest
import pytest

# Shield optional package imports
from ..optpkg import optional_package

from nibabel.pydicom_compat import have_dicom

PImage, have_pil, _ = optional_package('PIL.Image')

data_dir = pjoin(dirname(__file__), 'data')


def setUpModule():
    if os.name == 'nt':
        raise unittest.SkipTest('FUSE not available for windows, skipping dft tests')
    if not have_dicom:
        raise unittest.SkipTest('Need pydicom for dft tests, skipping')


def test_init():
    dft.clear_cache()
    dft.update_cache(data_dir)


def test_study():
    studies = dft.get_studies(data_dir)
    assert len(studies) == 1
    assert (studies[0].uid ==
                 '1.3.12.2.1107.5.2.32.35119.30000010011408520750000000022')
    assert studies[0].date == '20100114'
    assert studies[0].time == '121314.000000'
    assert studies[0].comments == 'dft study comments'
    assert studies[0].patient_name == 'dft patient name'
    assert studies[0].patient_id == '1234'
    assert studies[0].patient_birth_date == '19800102'
    assert studies[0].patient_sex == 'F'


def test_series():
    studies = dft.get_studies(data_dir)
    assert len(studies[0].series) == 1
    ser = studies[0].series[0]
    assert (ser.uid ==
                 '1.3.12.2.1107.5.2.32.35119.2010011420292594820699190.0.0.0')
    assert ser.number == '12'
    assert ser.description == 'CBU_DTI_64D_1A'
    assert ser.rows == 256
    assert ser.columns == 256
    assert ser.bits_allocated == 16
    assert ser.bits_stored == 12


def test_storage_instances():
    studies = dft.get_studies(data_dir)
    sis = studies[0].series[0].storage_instances
    assert len(sis) == 2
    assert sis[0].instance_number == 1
    assert sis[1].instance_number == 2
    assert (sis[0].uid ==
                 '1.3.12.2.1107.5.2.32.35119.2010011420300180088599504.0')
    assert (sis[1].uid ==
                 '1.3.12.2.1107.5.2.32.35119.2010011420300180088599504.1')


def test_storage_instance():
    pass


@unittest.skipUnless(have_pil, 'could not import PIL.Image')
def test_png():
    studies = dft.get_studies(data_dir)
    data = studies[0].series[0].as_png()
    im = PImage.open(BytesIO(data))
    assert im.size == (256, 256)


def test_nifti():
    studies = dft.get_studies(data_dir)
    data = studies[0].series[0].as_nifti()
    assert len(data) == 352 + 2 * 256 * 256 * 2
    h = nifti1.Nifti1Header(data[:348])
    assert h.get_data_shape() == (256, 256, 2)
