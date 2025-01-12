# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Helper utilities to be used in cmdline applications
"""

# global verbosity switch
import re

verbose_level = 0


def _err(msg=None):
    """To return a string to signal "error" in output table"""
    if msg is None:
        msg = 'error'
    return '!' + msg


def verbose(thing, msg):
    """Print `s` if `thing` is less than the `verbose_level`"""
    # TODO: consider using nibabel's logger
    if thing <= verbose_level:
        print(' ' * thing + msg)


def table2string(table, out=None):
    """Given list of lists figure out their common widths and print to out

    Parameters
    ----------
    table : list of lists of strings
      What is aimed to be printed
    out : None or stream
      Where to print. If None, return string

    Returns
    -------
    string if out was None
    """

    # equalize number of elements in each row
    nelements_max = len(table) and max(len(x) for x in table)

    table = [row + [''] * (nelements_max - len(row)) for row in table]
    for i, table_ in enumerate(table):
        table[i] += [''] * (nelements_max - len(table_))

    # eat whole entry while computing width for @w (for wide)
    markup_strip = re.compile('^@([lrc]|w.*)')
    col_width = [max(len(markup_strip.sub('', x)) for x in column) for column in zip(*table)]
    trans = str.maketrans("lrcw", "<>^^")
    lines = []
    for row in table:
        line = []
        for item, width in zip(row, col_width):
            item = str(item)
            if item.startswith('@'):
                align = item[1]
                item = item[2:]
                if align not in ('l', 'r', 'c', 'w'):
                    raise ValueError(f'Unknown alignment {align}. Known are l,r,c')
            else:
                align = 'c'

            line.append(f'{item:{align.translate(trans)}{width}}')
        lines.append(' '.join(line).rstrip())

    ret = '\n'.join(lines) + '\n'
    if out is not None:
        out.write(ret)
    else:
        return ret


def ap(helplist, format_, sep=', '):
    """Little helper to enforce consistency"""
    if helplist == '-':
        return helplist
    ls = [format_ % x for x in helplist]
    return sep.join(ls)


def safe_get(obj, name):
    """A getattr which would return '-' if getattr fails"""
    try:
        f = getattr(obj, 'get_' + name)
        return f()
    except Exception as e:
        verbose(2, f'get_{name}() failed -- {e}')
        return '-'
