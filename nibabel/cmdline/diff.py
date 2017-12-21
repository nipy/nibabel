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


def diff_values(key, inputs):
    diffs = []

    for i in range(len(inputs)):
        if i != len(inputs)-1:
            temp_input_1 = inputs[i]
            temp_input_2 = inputs[i+1]
            if np.any(temp_input_1[key] != temp_input_2[key]):
                if i != 0 and temp_input_2 != diffs:
                    diffs.append(temp_input_1[key])
                    diffs.append(temp_input_2[key])
                else:
                    diffs.append(temp_input_1[key])
            elif type(temp_input_1[key]) != type(temp_input_2[key]):
                if i != 0 and temp_input_2 != diffs:
                    diffs.append(temp_input_1[key])
                    diffs.append(temp_input_2[key])
                else:
                    diffs.append(temp_input_1[key])
            else:
                pass

    # TODO: figure out a way to not have these erroneous outputs occur in the above loop
    for a in range(len(diffs)-1):
        for b in range(len(diffs)-1):
            try:
                if np.all(np.isnan(diffs[a])) and np.all(np.isnan(diffs[a+1])):
                    del diffs[a]
            except TypeError:
                pass
            if a and b != len(diffs)-1:
                if np.any(diffs[a] == diffs[b]):
                    del diffs[a]

    if len(diffs) > 1:
        return {key: diffs}


def process_file(files, opts):

    file_list = []
    header_list = []

    for f in range(len(files)):
        file_list.append(nib.load(files[f]))
        for h in range(len(files)):
            header_list.append(file_list[f].header)

    if opts.header_fields:
        # signals "all fields"
        if opts.header_fields == 'all':
            # TODO: header fields might vary across file types, thus prior sensing would be needed
            header_fields = header_list[0].keys()
        else:
            header_fields = opts.header_fields.split(',')

        for f in header_fields:
            if diff_values(f, header_list) is not None:
                    print(diff_values(f, header_list))


def main():
    """NO DAYS OFF"""

    parser = get_opt_parser()
    (opts, files) = parser.parse_args()

    nibabel.cmdline.utils.verbose_level = opts.verbose

    assert len(files) >= 2, "Please enter at least two files"

    if nibabel.cmdline.utils.verbose_level < 3:
        # suppress nibabel format-compliance warnings
        nib.imageglobals.logger.level = 50

    process_file(files, opts)
