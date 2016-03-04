""" Utilities for working with DICOM datasets
"""

import re, string
from numpy.compat.py3k import asstr


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


def seconds_to_tm(seconds):
    '''Convert a float representing seconds past midnight into DICOM TM value

    Parameters
    ----------
    seconds : float
        Number of seconds past midnights

    Returns
    -------
    tm : str
        String suitable for use as value in DICOM element with VR of 'TM'
    '''
    hours = seconds // 3600
    seconds -= hours * 3600
    minutes = seconds // 60
    seconds -= minutes * 60
    res = '%02d%02d%08.5f' % (hours, minutes, seconds)
    return res


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


def years_to_as(years):
    '''Convert float representing age in years to DICOM 'AS' value

    Parameters
    ----------
    years : float
        The years of age

    Returns
    -------
    as : str
        String suitable for use as value in DICOM element with VR of 'AS'
    '''
    if years == round(years):
        return '%dY' % years

   # Choose how to represent the age (years, months, weeks, or days)
    conversions = (('Y', 1), ('M', 12), ('W', (365. / 7)), ('D', 365))
    # Try all the conversions, ignore ones that have more than three digits
    # which is the limit for the AS value representation, or where they round
    # to zero
    results = [(years * x[1], x[0]) for x in conversions]
    results = [x for x in results
               if round(x[0]) > 0 and len('%d' % x[0]) < 4]
    # Choose the first one that is close to the minimum error
    errors = [abs(x[0] - round(x[0])) for x in results]
    min_error = min(errors)
    best_idx = 0
    while errors[best_idx] - min_error > 0.001:
        best_idx += 1
    return '%d%s' % (round(results[best_idx][0]), results[best_idx][1])
