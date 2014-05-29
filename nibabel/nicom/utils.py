""" Utilities for working with DICOM datasets
"""
from __future__ import division, print_function, absolute_import

import os, uuid, hashlib
from random import random
from math import ceil

from ..py3k import asstr


def find_private_section(dcm_data, group_no, creator):
    """ Return start element in group `group_no` given creator name `creator`

    Private attribute tags need to announce where they will go by putting a tag
    in the private group (here `group_no`) between elements 1 and 0xFF.  The
    element number of these tags give the start of matching information, in the
    higher tag numbers.

    Paramters
    ---------
    dcm_data : dicom ``dataset``
        Iterating over `dcm_data` produces ``elements`` with attributes ``tag``,
        ``VR``, ``value``
    group_no : int
        Group number in which to search
    creator : str or bytes or regex
        Name of section - e.g. 'SIEMENS CSA HEADER' - or regex to search for
        section name.  Regex used via ``creator.search(element_value)`` where
        ``element_value`` is the value of the data element.

    Returns
    -------
    element_start : int
        Element number at which named section starts
    """
    is_regex = hasattr(creator, 'search')
    if not is_regex: # assume string / bytes
        creator = asstr(creator)
    for element in dcm_data: # Assumed ordered by tag (groupno, elno)
        grpno, elno = element.tag.group, element.tag.elem
        if grpno > group_no:
            break
        if grpno != group_no:
            continue
        if elno > 0xFF:
            break
        if element.VR not in ('LO', 'OB'):
            continue
        name = asstr(element.value).strip()
        if is_regex:
            if creator.search(name) != None:
                return elno * 0x100
        else: # string - needs exact match
            if creator == name:
                return elno * 0x100
    return None


def find_private_element(dcm_data, group_no, creator, elem_offset):
    """ Return the private element in group `group_no` given creator name
    `creator` and the offset for the element number `elem_offset`.

    Paramters
    ---------
    dcm_data : dicom ``dataset``
        Iterating over `dcm_data` produces ``elements`` with attributes ``tag``,
        ``VR``, ``value``
    group_no : int
        Group number in which to search
    creator : str or bytes or regex
        Name of section - e.g. 'SIEMENS CSA HEADER' - or regex to search for
        section name.  Regex used via ``creator.search(element_value)`` where
        ``element_value`` is the value of the data element.
    elem_offset : int
        The offset to the element from the start of the private section

    Returns
    -------
    element : dicom ``element``
        The private element or None if not found
    """
    if not 0 < elem_offset <= 0xFF:
        raise ValueError("The elem_offset is invalid")
    sect_start = find_private_section(dcm_data, group_no, creator)
    if sect_start is None:
        return None
    return dcm_data.get((group_no, sect_start + elem_offset))


def make_uid(entropy_srcs=None, prefix='2.25.'):
    '''Generate a DICOM UID value.

    Follows the advice given at:
    http://www.dclunie.com/medical-image-faq/html/part2.html#UID

    Parameters
    ----------
    entropy_srcs : list of str or None
        List of strings providing the entropy used to generate the UID. If
        None these will be collected from a combination of HW address, time,
        process ID, and randomness.
    '''
    # Combine all the entropy sources with a hashing algorithm
    if entropy_srcs is None:
        entropy_srcs = [str(uuid.uuid1()), # 128-bit from MAC/time/randomness
                        str(os.getpid()), # Current process ID
                        random().hex() # 64-bit randomness
                       ]
    hash_val = hashlib.sha256(''.join(entropy_srcs))

    # Converet this to an int with the maximum available digits
    avail_digits = 64 - len(prefix)
    int_val = int(hash_val.hexdigest(), 16) % (10 ** avail_digits)

    return prefix  + str(int_val)

def as_to_years(age_str):
    '''Take the value from a DICOM element with VR 'AS' and return the age
    in years as a float.'''
    age_str = age_str.strip()
    if age_str[-1] == 'Y':
        return float(age_str[:-1])
    elif age_str[-1] == 'M':
        return float(age_str[:-1]) / 12
    elif age_str[-1] == 'W':
        return float(age_str[:-1]) / 52.1775
    elif age_str[-1] == 'D':
        return float(age_str[:-1]) / 365
    else:
        return float(age_str[:-1])