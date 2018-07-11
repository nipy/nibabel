#!python
# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Quick summary of the differences among a set of neuroimaging files
"""
from __future__ import division, print_function, absolute_import

import re
import sys
from collections import OrderedDict
from optparse import OptionParser, Option

import numpy as np

import nibabel as nib
import nibabel.cmdline.utils
import hashlib


def get_opt_parser():
    # use module docstring for help output
    p = OptionParser(
        usage="%s [OPTIONS] [FILE ...]\n\n" % sys.argv[0] + __doc__,
        version="%prog " + nib.__version__)

    p.add_options([
        Option("-v", "--verbose", action="count",
               dest="verbose", default=0,
               help="Make more noise.  Could be specified multiple times"),

        Option("-H", "--header-fields",
               dest="header_fields", default='all',
               help="Header fields (comma separated) to be printed as well (if present)"),
    ])

    return p


def are_values_different(*values):
    """Generically compares values, returns true if different"""
    value0 = values[0]
    values = values[1:]  # to ensure that the first value isn't compared with itself

    for value in values:
        try:  # we sometimes don't want NaN values
            if np.any(np.isnan(value0)) and np.any(np.isnan(value)):  # if they're both NaN
                break
            elif np.any(np.isnan(value0)) or np.any(np.isnan(value)):  # if only 1 is NaN
                return True

        except TypeError:
            pass

        if type(value0) != type(value):  # if types are different, then we consider them different
            return True
        elif isinstance(value0, np.ndarray) and np.any(value0 != value):  # special test for ndarray
            return True
        elif value0 != value:
            return True

    return False


def get_headers_diff(file_headers, names=None):
    """Get difference between headers

    Parameters
    ----------
    file_headers: list of actual headers (dicts) from files
    names: list of header fields to test

    Returns
    -------
    dict
      str: list for each header field which differs, return list of
      values per each file
    """
    difference = OrderedDict()
    fields = names

    if names is None:
        fields = file_headers[0].keys()

    # for each header field
    for field in fields:
        values = [header.get(field) for header in file_headers]  # get corresponding value

        # if these values are different, store them in a dictionary
        if are_values_different(*values):
            difference[field] = values

    return difference


def get_data_md5sums(files):

    md5sums = [
        hashlib.md5(np.ascontiguousarray(nib.load(f).get_data(), dtype=np.float32)).hexdigest()
        for f in files
    ]

    if len(set(md5sums)) == 1:
        return []

    return md5sums


def main():
    """Getting the show on the road"""

    parser = get_opt_parser()
    (opts, files) = parser.parse_args()

    nibabel.cmdline.utils.verbose_level = opts.verbose

    assert len(files) >= 2, "Please enter at least two files"

    if nibabel.cmdline.utils.verbose_level < 3:
        # suppress nibabel format-compliance warnings
        nib.imageglobals.logger.level = 50

    file_headers = [nib.load(f).header for f in files]

    if opts.header_fields:  # will almost always have a header field
        # signals "all fields"
        if opts.header_fields == 'all':
            # TODO: header fields might vary across file types, thus prior sensing would be needed
            header_fields = file_headers[0].keys()
        else:
            header_fields = opts.header_fields.split(',')

    diff = get_headers_diff(file_headers, header_fields)
    data_diff = get_data_md5sums(files)

    if data_diff:
        diff['DATA(md5)'] = data_diff

    if diff:
        print("These files are different.")
        print("{:<15}".format('Field'), end="")

        for f in files:
            output = ""
            i = 0
            while i < len(f):
                if f[i] == "/" or f[i] == "\\":
                    output = ""
                else:
                    output += f[i]
                i += 1

            print("{:<55}".format(output), end="")

        print()

        for key, value in diff.items():
            print("{:<15}".format(key), end="")

            for item in value:
                item_str = str(item)
                # Value might start/end with some invisible spacing characters so we
                # would "condition" it on both ends a bit
                item_str = re.sub('^[ \t]+', '<', item_str)
                item_str = re.sub('[ \t]+$', '>', item_str)
                # and also replace some other invisible symbols with a question
                # mark
                item_str = re.sub('[\x00]', '?', item_str)
                print("{:<55}".format(item_str), end="")

            print()

        raise SystemExit(1)
    else:
        print("These files are identical.")
