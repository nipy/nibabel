""" Testing nicom.utils module
"""
import re

import numpy as np

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

from nose.tools import (assert_true, assert_false, assert_raises,
                        assert_equal, assert_not_equal)


from ..utils import (find_private_section, find_private_element, make_uid,
                     as_to_years, tm_to_seconds)

from .test_dicomwrappers import (have_dicom, dicom_test,
                                 IO_DATA_PATH, DATA, DATA_PHILIPS)


@dicom_test
def test_find_private_section_real():
    # Find section containing named private creator information
    # On real data first
    assert_equal(find_private_section(DATA, 0x29, 'SIEMENS CSA HEADER'),
                 0x1000)
    assert_equal(find_private_section(DATA, 0x29, 'SIEMENS MEDCOM HEADER2'),
                 0x1100)
    # These two work even though the private creator value in the dataset has
    # trailing whitespace
    assert_equal(find_private_section(DATA, 0x19, 'SIEMENS MR HEADER'),
                 0x1000)
    assert_equal(find_private_section(DATA, 0x51, 'SIEMENS MR HEADER'),
                 0x1000)
    # Return None when the section is not found
    assert_equal(find_private_section(DATA_PHILIPS, 0x29, 'SIEMENS CSA HEADER'),
                 None)
    # Make fake datasets
    from dicom.dataset import Dataset
    ds = Dataset({})
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


@dicom_test
def test_find_private_element():
    # Make sure ValueError is raised if the elem_offset is invalid
    assert_raises(ValueError,
                  find_private_element,
                  DATA,
                  0x19,
                  'SIEMENS MR HEADER',
                  -1
                 )
    assert_raises(ValueError,
                  find_private_element,
                  DATA,
                  0x19,
                  'SIEMENS MR HEADER',
                  0x100,
                 )

    # Find a specific private element
    assert_equal(find_private_element(DATA,
                                      0x19,
                                      'SIEMENS MR HEADER',
                                      0x8).value,
                 'IMAGE NUM 4 ')
    assert_equal(find_private_element(DATA,
                                      0x19,
                                      'SIEMENS MR HEADER',
                                      0xb).value,
                 '40')
    assert_equal(find_private_element(DATA,
                                      0x51,
                                      'SIEMENS MR HEADER',
                                      0xb).value,
                 '128p*128')


def test_make_uid():
    # Check that we get an exception if the prefix is too long
    assert_raises(ValueError, make_uid, None, '1'*64)
    # Make sure the result is 64 chars, regardless of prefix
    for prefix in ('1.', '1.2.', '1.' * 30):
        print prefix
        print len(make_uid(prefix=prefix))
        assert_true(len(make_uid(prefix=prefix)) == 64)
    # The same entropy inputs give the same results
    assert_equal(make_uid(['blah']), make_uid(['blah']))
    # Default entropy should give different results
    assert_not_equal(make_uid(), make_uid())


def test_as_to_years():
    assert_equal(as_to_years('1'), 1.0)
    assert_equal(as_to_years('1Y'), 1.0)
    assert_equal(as_to_years('53'), 53.0)
    assert_equal(as_to_years('53Y'), 53.0)
    assert_almost_equal(as_to_years('2M'), 2. / 12.)
    assert_almost_equal(as_to_years('2D'), 2. / 365.)
    assert_almost_equal(as_to_years('2W'), 2. * (7. / 365.))

def test_tm_to_seconds():
    for str_val in ('', '1', '111', '11111', '111111.', '1111111', '1:11',
                    ' 111'):
        assert_raises(ValueError, tm_to_seconds, str_val)
    assert_almost_equal(tm_to_seconds('01'), 60*60)
    assert_almost_equal(tm_to_seconds('0101'), 61*60)
    assert_almost_equal(tm_to_seconds('010101'), 61*60 + 1)
    assert_almost_equal(tm_to_seconds('010101.001'), 61*60 + 1.001)
    assert_almost_equal(tm_to_seconds('01:01:01.001'), 61*60 + 1.001)
