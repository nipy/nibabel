""" Testing nicom.utils module
"""
import re


from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


from ..utils import find_private_section

from nibabel.pydicom_compat import dicom_test, pydicom
from .test_dicomwrappers import (DATA, DATA_PHILIPS)


@dicom_test
def test_find_private_section_real():
    # Find section containing named private creator information
    # On real data first
    assert_equal(find_private_section(DATA, 0x29, 'SIEMENS CSA HEADER'),
                 0x1000)
    assert_equal(find_private_section(DATA, 0x29, 'SIEMENS MEDCOM HEADER2'),
                 0x1100)
    assert_equal(find_private_section(DATA_PHILIPS, 0x29, 'SIEMENS CSA HEADER'),
                 None)
    # Make fake datasets
    ds = pydicom.dataset.Dataset({})
    ds.add_new((0x11, 0x10), 'LO', b'some section')
    assert_equal(find_private_section(ds, 0x11, 'some section'), 0x1000)
    ds.add_new((0x11, 0x11), 'LO', b'anther section')
    ds.add_new((0x11, 0x12), 'LO', b'third section')
    assert_equal(find_private_section(ds, 0x11, 'third section'), 0x1200)
    # Wrong 'OB' is acceptable for VM (should be 'LO')
    ds.add_new((0x11, 0x12), 'OB', b'third section')
    assert_equal(find_private_section(ds, 0x11, 'third section'), 0x1200)
    # Anything else not acceptable
    ds.add_new((0x11, 0x12), 'PN', b'third section')
    assert_equal(find_private_section(ds, 0x11, 'third section'), None)
    # The input (DICOM value) can be a string insteal of bytes
    ds.add_new((0x11, 0x12), 'LO', 'third section')
    assert_equal(find_private_section(ds, 0x11, 'third section'), 0x1200)
    # Search can be bytes as well as string
    ds.add_new((0x11, 0x12), 'LO', b'third section')
    assert_equal(find_private_section(ds, 0x11, b'third section'), 0x1200)
    # Search with string or bytes must be exact
    assert_equal(find_private_section(ds, 0x11, b'third sectio'), None)
    assert_equal(find_private_section(ds, 0x11, 'hird sectio'), None)
    # The search can be a regexp
    assert_equal(find_private_section(ds,
                                      0x11,
                                      re.compile(r'third\Wsectio[nN]')),
                 0x1200)
    # No match -> None
    assert_equal(find_private_section(ds,
                                      0x11,
                                      re.compile(r'not third\Wsectio[nN]')),
                 None)
    # If there are gaps in the sequence before the one we want, that is OK
    ds.add_new((0x11, 0x13), 'LO', b'near section')
    assert_equal(find_private_section(ds, 0x11, 'near section'), 0x1300)
    ds.add_new((0x11, 0x15), 'LO', b'far section')
    assert_equal(find_private_section(ds, 0x11, 'far section'), 0x1500)
