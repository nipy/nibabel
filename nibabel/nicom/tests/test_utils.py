""" Testing nicom.utils module
"""
import re

import pytest
from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from ..utils import (find_private_section, seconds_to_tm, tm_to_seconds,
                     as_to_years, years_to_as)

from . import dicom_test
from ...pydicom_compat import pydicom
from .test_dicomwrappers import DATA, DATA_PHILIPS


@dicom_test
def test_find_private_section_real():
    # Find section containing named private creator information
    # On real data first
    assert find_private_section(DATA, 0x29, 'SIEMENS CSA HEADER') == 0x1000
    assert find_private_section(DATA, 0x29, 'SIEMENS MEDCOM HEADER2') == 0x1100
    assert find_private_section(DATA_PHILIPS, 0x29, 'SIEMENS CSA HEADER') == None
    # Make fake datasets
    ds = pydicom.dataset.Dataset({})
    ds.add_new((0x11, 0x10), 'LO', b'some section')
    assert find_private_section(ds, 0x11, 'some section') == 0x1000
    ds.add_new((0x11, 0x11), 'LO', b'anther section')
    ds.add_new((0x11, 0x12), 'LO', b'third section')
    assert find_private_section(ds, 0x11, 'third section') == 0x1200
    # Wrong 'OB' is acceptable for VM (should be 'LO')
    ds.add_new((0x11, 0x12), 'OB', b'third section')
    assert find_private_section(ds, 0x11, 'third section') == 0x1200
    # Anything else not acceptable
    ds.add_new((0x11, 0x12), 'PN', b'third section')
    assert find_private_section(ds, 0x11, 'third section') is None
    # The input (DICOM value) can be a string insteal of bytes
    ds.add_new((0x11, 0x12), 'LO', 'third section')
    assert find_private_section(ds, 0x11, 'third section') == 0x1200
    # Search can be bytes as well as string
    ds.add_new((0x11, 0x12), 'LO', b'third section')
    assert find_private_section(ds, 0x11, b'third section') == 0x1200
    # Search with string or bytes must be exact
    assert find_private_section(ds, 0x11, b'third sectio') is None
    assert find_private_section(ds, 0x11, 'hird sectio') is None
    # The search can be a regexp
    assert find_private_section(ds, 0x11, re.compile(r'third\Wsectio[nN]')) == 0x1200
    # No match -> None
    assert find_private_section(ds, 0x11, re.compile(r'not third\Wsectio[nN]')) is None
    # If there are gaps in the sequence before the one we want, that is OK
    ds.add_new((0x11, 0x13), 'LO', b'near section')
    assert find_private_section(ds, 0x11, 'near section') == 0x1300
    ds.add_new((0x11, 0x15), 'LO', b'far section')
    assert find_private_section(ds, 0x11, 'far section') == 0x1500


def test_tm_to_seconds():
    for str_val in ('', '1', '111', '11111', '111111.', '1111111', '1:11',
                    ' 111'):
        with pytest.raises(ValueError):
            tm_to_seconds(str_val)
    assert_almost_equal(tm_to_seconds('01'), 60*60)
    assert_almost_equal(tm_to_seconds('0101'), 61*60)
    assert_almost_equal(tm_to_seconds('010101'), 61*60 + 1)
    assert_almost_equal(tm_to_seconds('010101.001'), 61*60 + 1.001)
    assert_almost_equal(tm_to_seconds('01:01:01.001'), 61*60 + 1.001)
    assert_almost_equal(tm_to_seconds('02:03'), 123 * 60)


def test_tm_rt():
    for tm_val in ('010101.00000', '010101.00100', '122432.12345'):
        assert tm_val == seconds_to_tm(tm_to_seconds(tm_val))


def test_as_to_years():
    assert as_to_years('1') == 1.0
    assert as_to_years('1Y') == 1.0
    assert as_to_years('53') == 53.0
    assert as_to_years('53Y') == 53.0
    assert_almost_equal(as_to_years('2M'), 2. / 12.)
    assert_almost_equal(as_to_years('2D'), 2. / 365.)
    assert_almost_equal(as_to_years('2W'), 2. * (7. / 365.))


def test_as_rt():
    # Round trip
    for as_val in ('1Y', '53Y', '153Y',
                   '2M', '42M', '200M',
                   '2W', '42W', '930W',
                   '2D', '45D', '999D'):
        assert as_val == years_to_as(as_to_years(as_val))
    # Any day multiple of 7 may be represented as weeks
    for as_val, other_as_val in (('7D', '1W'),
                                 ('14D', '2W'),
                                 ('21D', '3W'),
                                 ('42D', '6W')):
        assert years_to_as(as_to_years(as_val)) in (as_val, other_as_val)
