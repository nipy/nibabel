# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Parse the "ASCCONV" meta data format found in a variety of Siemens MR files.
"""
from ..externals import OrderedDict


class AscconvParseError(Exception):
    def __init__(self, line):
        '''Exception indicating a error parsing a line from the ASCCONV
        format.
        '''
        self.line = line

    def __str__(self):
        return 'Unable to parse ASCCONV line: %s' % self.line


def _parse_ascconv_line(line, str_delim='""'):
    delim_len = len(str_delim)

    # Lines can have comments at the tail end, denoted with a '#' symbol
    comment_idx = line.find('#')
    if comment_idx != -1:
        # Check if the pound sign is in a string literal
        if line[:comment_idx].count(str_delim) == 1:
            if line[comment_idx:].find(str_delim) == -1:
                raise AscconvParseError(line)
        else:
            line = line[:comment_idx]

    # Allow empty lines
    if line.strip() == '':
        return None

    # Find the first equals sign and use that to split key/value
    equals_idx = line.find('=')
    if equals_idx == -1:
        raise AscconvParseError(line)
    key = line[:equals_idx].strip()
    val_str = line[equals_idx + 1:].strip()

    # If there is a string literal, pull that out
    if val_str.startswith(str_delim):
        end_quote = val_str[delim_len:].find(str_delim) + delim_len
        if end_quote == -1:
            raise AscconvParseError(line)
        elif not end_quote == len(val_str) - delim_len:
            # Make sure remainder is just comment
            if not val_str[end_quote+delim_len:].strip().startswith('#'):
                raise AscconvParseError(line)

        return (key, val_str[2:end_quote])

    else: # Otherwise try to convert to an int or float
        val = None
        try:
            val = int(val_str)
        except ValueError:
            pass
        else:
            return (key, val)

        try:
            val = int(val_str, 16)
        except ValueError:
            pass
        else:
            return (key, val)

        try:
            val = float(val_str)
        except ValueError:
            pass
        else:
            return (key, val)

    raise AscconvParseError(line)


def parse_ascconv(csa_key, input_str):
    '''Parse the 'ASCCONV' format from `input_str`.

    Parameters
    ----------
    csa_key : str
        The key in the CSA dict for the element containing `input_str`. Should
        be 'MrPheonixProtocol' or 'MrProtocol'.

    input_str: str
        The string we are parsing

    Returns
    -------
    prot_dict : OrderedDict
        Meta data pulled from the ASCCONV section.

    Raises
    ------
    AscconvParseError : A line of the ASCCONV section could not be parsed.
    '''
    if csa_key == 'MrPhoenixProtocol':
        str_delim = '""'
    elif csa_key == 'MrProtocol':
        str_delim = '"'
    else:
        raise ValueError('Unknown protocol key: %s' % csa_key)
    ascconv_start = input_str.find('### ASCCONV BEGIN ###')
    ascconv_end = input_str.find('### ASCCONV END ###')
    ascconv_lines = input_str[ascconv_start:ascconv_end].split('\n')[1:-1]

    result = OrderedDict()
    for line in ascconv_lines:
        parse_result = _parse_ascconv_line(line, str_delim)
        if parse_result:
            result[parse_result[0]] = parse_result[1]

    return result
