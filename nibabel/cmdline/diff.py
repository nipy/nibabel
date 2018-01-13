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

import sys
from optparse import OptionParser, Option

import numpy as np

import nibabel as nib
import nibabel.cmdline.utils
from nibabel.cmdline.utils import _err, verbose, table2string, ap, safe_get
import fileinput

__author__ = 'Yaroslav Halchenko & Christopher Cheng'
__copyright__ = 'Copyright (c) 2017 NiBabel contributors'
__license__ = 'MIT'


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
    if np.any(compare1 != compare2):
        return compare1, compare2
    elif type(compare1) != type(compare2):
        return compare1, compare2
    else:
        pass


def diff_header_fields(key, inputs):
    """Iterates over a single header field of multiple files"""
    diffs = []

    if len(inputs) > 2:
        for input_1, input_2 in zip(inputs, inputs[1:]):
                key_1 = input_1[key]
                key_2 = input_2[key]

                if diff_values(key_1, key_2):
                    if input_1 != inputs[0] and input_2 not in diffs:
                        diffs.append(key_1)
                        diffs.append(key_2)
                    else:
                        diffs.append(key_1)

    else:
        input_1 = inputs[0]
        input_2 = inputs[1]

        key_1 = input_1[key]
        key_2 = input_2[key]

        if np.any(key_1 != key_2):
            diffs.append(key_1)
            diffs.append(key_2)
        elif type(key_1) != type(key_2):
            diffs.append(key_1)
            diffs.append(key_2)

    # TODO: figure out a way to not have these erroneous outputs occur in the above loop
    for a in range(len(diffs)-1):
        for b in range(len(diffs)-1):
            try:
                if np.all(np.isnan(diffs[a])) and np.all(np.isnan(diffs[a+1])):
                    del diffs[a]
            except TypeError:
                pass

    if len(diffs) > 1:
        return {key: diffs}


def get_headers_diff(files, opts):

    header_list = [nib.load(f).header for f in files]

    if opts.header_fields:
        # signals "all fields"
        if opts.header_fields == 'all':
            # TODO: header fields might vary across file types, thus prior sensing would be needed
            header_fields = header_list[0].keys()
        else:
            header_fields = opts.header_fields.split(',')

        for f in header_fields:
            if diff_header_fields(f, header_list) is not None:
                    print(diff_header_fields(f, header_list))


def main():
    """NO DAYS OFF"""

    parser = get_opt_parser()
    (opts, files) = parser.parse_args()

    nibabel.cmdline.utils.verbose_level = opts.verbose

    assert len(files) >= 2, "Please enter at least two files"

    if nibabel.cmdline.utils.verbose_level < 3:
        # suppress nibabel format-compliance warnings
        nib.imageglobals.logger.level = 50

    get_headers_diff(files, opts)
