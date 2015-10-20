""" Utilities for working with DICOM datasets
"""
from __future__ import division, print_function, absolute_import

from ..py3k import asstr


def find_private_section(dcm_data, group_no, creator):
    """ Return start element in group `group_no` given creator name `creator`

    Private attribute tags need to announce where they will go by putting a tag
    in the private group (here `group_no`) between elements 1 and 0xFF.  The
    element number of these tags give the start of matching information, in the
    higher tag numbers.

    Parameters
    ----------
    dcm_data : dicom ``dataset``
        Iterating over `dcm_data` produces ``elements`` with attributes
        ``tag``, ``VR``, ``value``
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
    if not is_regex:  # assume string / bytes
        creator = asstr(creator)
    for element in dcm_data:  # Assumed ordered by tag (groupno, elno)
        grpno, elno = element.tag.group, element.tag.elem
        if grpno > group_no:
            break
        if grpno != group_no:
            continue
        if elno > 0xFF:
            break
        if element.VR not in ('LO', 'OB'):
            continue
        name = asstr(element.value)
        if is_regex:
            if creator.search(name) is not None:
                return elno * 0x100
        else:  # string - needs exact match
            if creator == name:
                return elno * 0x100
    return None
