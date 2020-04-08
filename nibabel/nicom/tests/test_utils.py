""" Testing nicom.utils module
"""
import re

from ..utils import find_private_section

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
