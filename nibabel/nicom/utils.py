""" Utilities for working with DICOM datasets
"""
from __future__ import division, print_function, absolute_import

import os, uuid, hashlib, re, string
from random import random

from ..py3k import asstr, asbytes


def find_private_section(dcm_data, group_no, creator):
    """ Return start element in group `group_no` given creator name `creator`

    Private attribute tags need to announce where they will go by putting a tag
    in the private group (here `group_no`) between elements 1 and 0xFF.  The
    element number of these tags give the start of matching information, in the
    higher tag numbers.

    Parameters
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

    Parameters
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


def as_to_years(age_str):
    '''Convert a DICOM age value (value representation of 'AS') to the age in
    years.

    Parameters
    ----------
    age_str : str
        The string value from the DICOM element

    Returns
    -------
    age : float
        The age of the subject in years
    '''
    age_str = age_str.strip()
    if age_str[-1] == 'Y':
        return float(age_str[:-1])
    elif age_str[-1] == 'M':
        return float(age_str[:-1]) / 12
    elif age_str[-1] == 'W':
        return float(age_str[:-1]) / (365. / 7)
    elif age_str[-1] == 'D':
        return float(age_str[:-1]) / 365
    else:
        return float(age_str)


def tm_to_seconds(time_str):
    '''Convert a DICOM time value (value representation of 'TM') to the number
    of seconds past midnight.

    Parameters
    ----------
    time_str : str
        The string value from the DICOM element

    Returns
    -------
    sec_past_midnight : float
        The number of seconds past midnight
    '''
    # Allow trailing white space
    time_str = time_str.rstrip()

    # Allow ACR/NEMA style format which includes colons between hours/minutes
    # and minutes/seconds
    colons = [x.start() for x in re.finditer(':', time_str)]
    if len(colons) > 0:
        if colons not in ([2], [2, 5]):
            raise ValueError("Invalid use of colons in 'TM' VR")
        time_str = time_str.replace(':', '')

    # Make sure the string length is valid
    str_len = len(time_str)
    is_valid = str_len > 0
    if str_len <= 6:
        # If there are six or less chars, there should be an even number
        if str_len % 2 != 0:
            is_valid = False
    else:
        # If there are more than six chars, the seventh position should be
        # a decimal followed by at least one digit
        if str_len == 7 or time_str[6] != '.':
            is_valid = False
    if not is_valid:
        raise ValueError("Invalid number of digits for 'TM' VR")

    # Make sure we don't have leading white space
    if time_str[0] in string.whitespace:
        raise ValueError("Leading whitespace not allowed in 'TM' VR")

    # The minutes and seconds are optional
    result = int(time_str[:2]) * 3600
    if str_len > 2:
        result += int(time_str[2:4]) * 60
    if str_len > 4:
        result += float(time_str[4:])

    return float(result)
