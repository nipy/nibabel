# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Parse the "Phoenix" meta data format found in a variety of Siemens MR files.
"""
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict


class PhoenixParseError(Exception):
    def __init__(self, line):
        '''Exception indicating a error parsing a line from the Phoenix
        Protocol.
        '''
        self.line = line

    def __str__(self):
        return 'Unable to parse phoenix protocol line: %s' % self.line


def _parse_phoenix_line(line, str_delim='""'):
    delim_len = len(str_delim)
    # Handle most comments (not always when string literal involved)
    comment_idx = line.find('#')
    if comment_idx != -1:
        # Check if the pound sign is in a string literal
        if line[:comment_idx].count(str_delim) == 1:
            if line[comment_idx:].find(str_delim) == -1:
                raise PhoenixParseError(line)
        else:
            line = line[:comment_idx]

    # Allow empty lines
    if line.strip() == '':
        return None

    # Find the first equals sign and use that to split key/value
    equals_idx = line.find('=')
    if equals_idx == -1:
        raise PhoenixParseError(line)
    key = line[:equals_idx].strip()
    val_str = line[equals_idx + 1:].strip()

    # If there is a string literal, pull that out
    if val_str.startswith(str_delim):
        end_quote = val_str[delim_len:].find(str_delim) + delim_len
        if end_quote == -1:
            raise PhoenixParseError(line)
        elif not end_quote == len(val_str) - delim_len:
            # Make sure remainder is just comment
            if not val_str[end_quote+delim_len:].strip().startswith('#'):
                raise PhoenixParseError(line)

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

    raise PhoenixParseError(line)


def parse_phoenix_prot(prot_key, prot_val):
    '''Parse the MrPheonixProtocol string.

    Parameters
    ----------
    prot_str : str
        The 'MrPheonixProtocol' string from the CSA Series sub header.

    Returns
    -------
    prot_dict : OrderedDict
        Meta data pulled from the ASCCONV section.

    Raises
    ------
    PhoenixParseError : A line of the ASCCONV section could not be parsed.
    '''
    if prot_key == 'MrPhoenixProtocol':
        str_delim = '""'
    elif prot_key == 'MrProtocol':
        str_delim = '"'
    else:
        raise ValueError('Unknown protocol key: %s' % prot_key)
    ascconv_start = prot_val.find('### ASCCONV BEGIN ###')
    ascconv_end = prot_val.find('### ASCCONV END ###')
    ascconv = prot_val[ascconv_start:ascconv_end].split('\n')[1:-1]

    result = OrderedDict()
    for line in ascconv:
        parse_result = _parse_phoenix_line(line, str_delim)
        if parse_result:
            result[parse_result[0]] = parse_result[1]

    return result
