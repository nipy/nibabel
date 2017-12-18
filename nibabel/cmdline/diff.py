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
Output a summary table for neuroimaging files (resolution, dimensionality, etc.)
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


def diff_values(key, compare1, compare2):
    """Returns the differences between two dicts"""
    if np.any(compare1 != compare2):
        return {key: (compare1, compare2)}
    elif type(compare1) != type(compare2):
        return {key: (compare1, compare2)}
    else:
        pass


def proc_file(f1, f2, opts):

    vol = nib.load(f1)
    vol2 = nib.load(f2)
    h = vol.header
    h2 = vol2.header

    if opts.header_fields:
        # signals "all fields"
        if opts.header_fields == 'all':
            # TODO: might vary across file types, thus prior sensing
            # would be needed
            header_fields = h.keys()
        else:
            header_fields = opts.header_fields.split(',')

        for f in header_fields:
            # if not f:  # skip empty
            #    continue
            if diff_values(f, h[f], h2[f]) is not None:
                print(diff_values(f, h[f], h2[f]))


def main():
    """Show must go on"""

    parser = get_opt_parser()
    (opts, files) = parser.parse_args()

    nibabel.cmdline.utils.verbose_level = opts.verbose

    assert len(files) == 2, "Please enter two files"
    # TODO #3 -- make it work for any number

    if nibabel.cmdline.utils.verbose_level < 3:
        # suppress nibabel format-compliance warnings
        nib.imageglobals.logger.level = 50

    rows = [proc_file(files[0], files[1], opts)]

    print(rows)
