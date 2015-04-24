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
import ast
from ..externals import OrderedDict


def parse_ascconv(csa_key, ascconv_str):
    '''Parse the 'ASCCONV' format from `input_str`.

    Parameters
    ----------
    csa_key : str
        The key in the CSA dict for the element containing `input_str`. Should
        be 'MrPheonixProtocol' or 'MrProtocol'.
    input_str : str
        The string we are parsing

    Returns
    -------
    prot_dict : OrderedDict
        Meta data pulled from the ASCCONV section.

    Raises
    ------
    SyntaxError
        A line of the ASCCONV section could not be parsed.
    '''
    if csa_key == 'MrPhoenixProtocol':
        str_delim = '""'
    elif csa_key == 'MrProtocol':
        str_delim = '"'
    else:
        raise ValueError('Unknown protocol key: %s' % csa_key)
    # Normalize string start / end markers to something Python understands
    ascconv_str = ascconv_str.replace(str_delim, '"""')
    ascconv_lines = ascconv_str.split('\n')
    # Use Python's own parser to parse modified ASCCONV assignments
    tree = ast.parse(ascconv_str)

    result = OrderedDict()
    for statement in tree.body:
        assert isinstance(statement, ast.Assign)
        value = ast.literal_eval(statement.value)
        # Get LHS string from corresponding text line
        key = ascconv_lines[statement.lineno - 1].split('=')[0].strip()
        result[key] = value

    return result
