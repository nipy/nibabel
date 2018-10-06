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
import os


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
    """Generically compare values, return True if different

    Note that comparison is targetting reporting of comparison of the headers
    so has following specifics:
    - even a difference in data types is considered a difference, i.e. 1 != 1.0
    - NaNs are considered to be the "same", although generally NaN != NaN  
    """
    value0 = values[0]

    # to not recompute over again
    if isinstance(value0, np.ndarray):
        value0_nans = np.isnan(value0)
        if not np.any(value0_nans):
            value0_nans = None

    for value in values[1:]:
        if type(value0) != type(value):  # if types are different, then we consider them different
            return True
        elif isinstance(value0, np.ndarray):
            if value0.dtype != value.dtype or \
               value0.shape != value.shape:
                return True
            # there might be NaNs and they need special treatment
            if value0_nans is not None:
                value_nans = np.isnan(value)
                if np.any(value0_nans != value_nans):
                    return True
                if np.any(value0[np.logical_not(value0_nans)]
                          != value[np.logical_not(value0_nans)]):
                    return True
            elif np.any(value0 != value):
                return True
        elif value0 is np.NaN:
            if value is not np.NaN:
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


def get_data_diff(files):
    """Get difference between md5 values

        Parameters
        ----------
        files: list of actual files

        Returns
        -------
        list
          np.array: md5 values of respective files
        """

    md5sums = [
        hashlib.md5(np.ascontiguousarray(nib.load(f).get_data(), dtype=np.float32)).hexdigest()
        for f in files
    ]

    if len(set(md5sums)) == 1:
        return []

    return md5sums


def display_diff(files, diff):
    """Format header differences into a nice string

        Parameters
        ----------
        files: list of files that were compared so we can print their names
        diff: dict of different valued header fields

        Returns
        -------
        str
          string-formatted table of differences
    """
    output = ""
    field_width = "{:<15}"
    value_width = "{:<55}"

    output += "These files are different.\n"
    output += field_width.format('Field')

    for f in files:
        output += value_width.format(os.path.basename(f))

    output += "\n"

    for key, value in diff.items():
        output += field_width.format(key)

        for item in value:
            item_str = str(item)
            # Value might start/end with some invisible spacing characters so we
            # would "condition" it on both ends a bit
            item_str = re.sub('^[ \t]+', '<', item_str)
            item_str = re.sub('[ \t]+$', '>', item_str)
            # and also replace some other invisible symbols with a question
            # mark
            item_str = re.sub('[\x00]', '?', item_str)
            output += value_width.format(item_str)

        output += "\n"

    return output


def main(args=None, out=None):
    """Getting the show on the road"""
    out = out or sys.stdout
    parser = get_opt_parser()
    (opts, files) = parser.parse_args(args)

    nibabel.cmdline.utils.verbose_level = opts.verbose

    if nibabel.cmdline.utils.verbose_level < 3:
        # suppress nibabel format-compliance warnings
        nib.imageglobals.logger.level = 50

    assert len(files) >= 2, "Please enter at least two files"

    file_headers = [nib.load(f).header for f in files]

    # signals "all fields"
    if opts.header_fields == 'all':
        # TODO: header fields might vary across file types, thus prior sensing would be needed
        header_fields = file_headers[0].keys()
    else:
        header_fields = opts.header_fields.split(',')

    diff = get_headers_diff(file_headers, header_fields)
    data_diff = get_data_diff(files)

    if data_diff:
        diff['DATA(md5)'] = data_diff

    if diff:
        out.write(display_diff(files, diff))
        raise SystemExit(1)

    else:
        out.write("These files are identical.\n")
        raise SystemExit(0)
