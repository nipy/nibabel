""" Utilities for working with DICOM datasets
"""

import re, string

from numpy.compat.py3k import asstr
import numpy as np

from ..externals import OrderedDict


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


TM_EXP = re.compile(r"^(\d\d)(\d\d)?(\d\d)?(\.\d+)?$")
# Allow ACR/NEMA style format which includes colons between hours/minutes and
# minutes/seconds.  See TM / time description in PS3.5 of the DICOM standard at
# http://dicom.nema.org/Dicom/2011/11_05pu.pdf
TM_EXP_1COLON = re.compile(r"^(\d\d):(\d\d)()?()?$")
TM_EXP_2COLONS = re.compile(r"^(\d\d):(\d\d):(\d\d)?(\.\d+)?$")


def tm_to_seconds(time_str):
    '''Convert DICOM time value (VR of 'TM') to seconds past midnight.

    Parameters
    ----------
    time_str : str
        The string value from the DICOM element

    Returns
    -------
    sec_past_midnight : float
        The number of seconds past midnight

    Notes
    -----
    From TM / time description in `PS3.5 of the DICOM standard
    <http://dicom.nema.org/Dicom/2011/11_05pu.pdf>`_::

        A string of characters of the format HHMMSS.FFFFFF; where HH contains
        hours (range "00" - "23"), MM contains minutes (range "00" - "59"), SS
        contains seconds (range "00" - "60"), and FFFFFF contains a fractional
        part of a second as small as 1 millionth of a second (range “000000” -
        “999999”). A 24-hour clock is used. Midnight shall be represented by
        only “0000“ since “2400“ would violate the hour range. The string may
        be padded with trailing spaces.  Leading and embedded spaces are not
        allowed.

        One or more of the components MM, SS, or FFFFFF may be unspecified as
        long as every component to the right of an unspecified component is
        also unspecified, which indicates that the value is not precise to the
        precision of those unspecified components.
    '''
    # Allow trailing white space
    time_str = time_str.rstrip()
    for matcher in (TM_EXP, TM_EXP_1COLON, TM_EXP_2COLONS):
        match = matcher.match(time_str)
        if match is not None:
            break
    else:
        raise ValueError('Invalid tm string "{0}"'.format(time_str))
    parts = [float(v) if v else 0 for v in match.groups()]
    return np.multiply(parts, [3600, 60, 1, 1]).sum()


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

    Notes
    -----
    See docstring for :func:`tm_to_seconds`.
    '''
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return '%02d%02d%08.5f' % (hours, minutes, seconds)


CONVERSIONS = OrderedDict((('Y', 1), ('M', 12), ('W', (365. / 7)), ('D', 365)))
CONV_KEYS = list(CONVERSIONS)
CONV_VALS = np.array(list(CONVERSIONS.values()))

AGE_EXP = re.compile(r'^(\d+)(Y|M|W|D)?$')


def as_to_years(age_str):
    '''Convert DICOM age value (VR of 'AS') to the age in years

    Parameters
    ----------
    age_str : str
        The string value from the DICOM element

    Returns
    -------
    age : float
        The age of the subject in years

    Notes
    -----
    From AS / age string description in `PS3.5 of the DICOM standard
    <http://dicom.nema.org/Dicom/2011/11_05pu.pdf>`_::

        A string of characters with one of the following formats -- nnnD, nnnW,
        nnnM, nnnY; where nnn shall contain the number of days for D, weeks for
        W, months for M, or years for Y.  Example: “018M” would represent an
        age of 18 months.
    '''
    match = AGE_EXP.match(age_str.strip())
    if not match:
        raise ValueError('Invalid age string "{0}"'.format(age_str))
    val, code = match.groups()
    code = 'Y' if code is None else code
    return float(val) / CONVERSIONS[code]


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

    Notes
    -----
    See docstring for :func:`as_to_years`.
    '''
    if years == round(years):
        return '%dY' % years
    # Choose how to represent the age (years, months, weeks, or days).
    # Try all the conversions, ignore ones that have more than three digits,
    # which is the limit for the AS value representation.
    conved = years * CONV_VALS
    conved[conved >= 1000] = np.nan  # Too many digits for AS field
    year_error = np.abs(conved - np.round(conved)) / CONV_VALS
    best_i = np.nanargmin(year_error)
    return "{0:.0f}{1}".format(conved[best_i], CONV_KEYS[best_i])
