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
               dest="header_fields", default='',
               help="Header fields (comma separated) to be printed as well (if present)"),
    ])

    return p


def diff_dicts(key, compare1, compare2):
    """Returns the differences between two dicts"""
    if np.any(compare1 != compare2):
        return {key: (compare1, compare2)}
    elif type(compare1) != type(compare2):
        return {key: (compare1, compare2)}
    else:
        pass


def main():
    """Show must go on"""

    parser = get_opt_parser()
    (opts, files) = parser.parse_args()

    nibabel.cmdline.utils.verbose_level = opts.verbose

    if nibabel.cmdline.utils.verbose_level < 3:
        # suppress nibabel format-compliance warnings
        nib.imageglobals.logger.level = 50

    assert len(files) == 2, "Please enter two files"
    # TODO #3 -- make it work for any number

    img1 = nib.load(files[0])
    img2 = nib.load(files[1])

    for i in img1.header.keys():
        if diff_dicts(i, img1.header[i], img2.header[i]) is not None:
            print(diff_dicts(i, img1.header[i], img2.header[i]))

    #  TODO #2 -- limit comparison only to certain fields
