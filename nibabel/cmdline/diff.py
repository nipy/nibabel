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
import itertools
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


def diff_values(compare1, compare2):
    """Generically compares two values, returns true if different"""
    if np.any(compare1 != compare2):
        return True
    elif type(compare1) != type(compare2):
        return True
    else:
        return compare1 != compare2


def diff_header_fields(key, inputs):
    """Iterates over a single header field of multiple files"""

    keyed_inputs = []

    for i in inputs:  # stores each file's respective header files
        try:
            field_value = i[key]
        except ValueError:
            continue

        try:  # filter numpy arrays
            if np.all(np.isnan(field_value)):
                continue
        except TypeError:
            pass

        for x in inputs[1:]:  # compare different values, print all as soon as diff is found
            try:
                data_diff = diff_values(str(x[key].dtype), str(field_value.dtype))

                if data_diff:
                    break
            except ValueError:
                continue

        try:
            if data_diff:  # prints data types if they're different and not if they're not
                if field_value.ndim < 1:
                    keyed_inputs.append("{}@{}".format(field_value, field_value.dtype))
                elif field_value.ndim == 1:
                    keyed_inputs.append("{}@{}".format(list(field_value), field_value.dtype))
            else:
                if field_value.ndim < 1:
                    keyed_inputs.append("{}".format(field_value))
                elif field_value.ndim == 1:
                    keyed_inputs.append("{}".format(list(field_value)))
        except UnboundLocalError:
            continue

    if keyed_inputs:  # sometimes keyed_inputs is empty lol
        comparison_input = keyed_inputs[0]

        for i in keyed_inputs[1:]:
            if diff_values(comparison_input, i):
                return keyed_inputs


def get_headers_diff(files, opts):
    """Get difference between headers

    Parameters
    ----------
    files: list of files
    opts: any options included from the command line

    Returns
    -------
    dict
      str: list  for each header field which differs, return list of
      values per each file
    """

    header_list = [nib.load(f).header for f in files]
    output = OrderedDict()

    if opts.header_fields:  # will almost always have a header field
        # signals "all fields"
        if opts.header_fields == 'all':
            # TODO: header fields might vary across file types, thus prior sensing would be needed
            header_fields = header_list[0].keys()
        else:
            header_fields = opts.header_fields.split(',')

        for f in header_fields:
            val = diff_header_fields(f, header_list)

            if val:
                output[f] = val

    return output


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

    diff = get_headers_diff(files, opts)
    data_diff = get_data_md5sums(files)
    if data_diff:
        diff['DATA(md5)'] = data_diff

    if diff:
        print("These files are different.")
        print("{:<11}".format('Field'), end="")

        for f in files:
            output = ""
            i = 0
            while i < len(f):
                if f[i] == "/" or f[i] == "\\":
                    output = ""
                else:
                    output += f[i]
                i += 1

            print("{:<45}".format(output), end="")

        print()

        for key, value in diff.items():
            print("{:<11}".format(key), end="")

            for item in value:
                item_str = str(item)
                # Value might start/end with some invisible spacing characters so we
                # would "condition" it on both ends a bit
                item_str = re.sub('^[ \t]+', '<', item_str)
                item_str = re.sub('[ \t]+$', '>', item_str)
                # and also replace some other invisible symbols with a question
                # mark
                item_str = re.sub('[\x00]', '?', item_str)
                print("{:<45}".format(item_str), end="")

            print()

        raise SystemExit(1)
    else:
        print("These files are identical.")
