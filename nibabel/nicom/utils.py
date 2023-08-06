"""Utilities for working with DICOM datasets
"""


def find_private_section(dcm_data, group_no, creator):
    """Return start element in group `group_no` given creator name `creator`

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
        Element number at which named section starts.
    """
    if hasattr(creator, 'search'):
        match_func = creator.search
    else:
        if isinstance(creator, bytes):
            creator = creator.decode('latin-1')
        match_func = creator.__eq__
    # Group elements assumed ordered by tag (groupno, elno)
    for element in dcm_data.group_dataset(group_no):
        elno = element.tag.elem
        if elno > 0xFF:
            break
        if element.VR not in ('LO', 'OB'):
            continue
        val = element.value
        if isinstance(val, bytes):
            val = val.decode('latin-1')
        if match_func(val):
            return elno * 0x100
    return None
