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
from six import binary_type

import numpy as np

import json_tricks
import yaml

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

        Option("-t", "--text",
               dest="text",
               help="Print output in a very nice-looking way"),

        Option("-j", "--json",
               dest="json",
               help="Print output in a json way"),

        Option("-y", "--yaml",
               dest="yaml",
               help="Print output in a yaml way"),
    ])

    return p


def diff_values(compare1, compare2):
    """Generically compares two values, returns true if different"""
    if np.any(compare1 != compare2):
        return True
    elif type(compare1) != type(compare2):
        return True
    else:
        return False


def diff_header_fields(key, inputs):
    """Iterates over a single header field of multiple files"""

    keyed_inputs = []

    for i in inputs:  # stores each file's respective header files
        if isinstance(i[key], (bytes, bytearray)):
            i[key] = str(i[key][2:])

        try:
            if np.all(np.isnan(i[key])):
                continue
        except TypeError:
            pass

        keyed_inputs.append(i[key])

    if keyed_inputs:  # sometimes keyed_inputs is empty lol
        comparison_input = keyed_inputs[0]

        for i in keyed_inputs[1:]:
            if diff_values(comparison_input, i):
                return keyed_inputs


def get_headers_diff(files, opts):

    header_list = [nib.load(f).header for f in files]

    if opts.header_fields:
        # signals "all fields"
        if opts.header_fields == 'all':
            # TODO: header fields might vary across file types, thus prior sensing would be needed
            header_fields = header_list[0].keys()
        else:
            header_fields = opts.header_fields.split(',')

        output = {}

        for f in header_fields:
            val = diff_header_fields(f, header_list)

            if val:
                output[f] = val

        return output


def main():
    """NO DAYS OFF"""

    parser = get_opt_parser()
    (opts, files) = parser.parse_args()

    nibabel.cmdline.utils.verbose_level = opts.verbose

    assert len(files) >= 2, "Please enter at least two files"

    if nibabel.cmdline.utils.verbose_level < 3:
        # suppress nibabel format-compliance warnings
        nib.imageglobals.logger.level = 50

    diff = get_headers_diff(files, opts)

    if opts.text:
        for x in diff:
            print(x+":", diff[x])

    # elif opts.json:
    #     print(json_tricks.dumps(diff, conv_str_byte=True))
    #
    # elif opts.yaml:
    #     print(yaml.dump(diff))
