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


def main():
    """Show must go on"""

    parser = get_opt_parser()
    (opts, files) = parser.parse_args()

    nibabel.cmdline.utils.verbose_level = opts.verbose

    if nibabel.cmdline.utils.verbose_level < 3:
        # suppress nibabel format-compliance warnings
        nib.imageglobals.logger.level = 50

    assert len(files) == 2, "ATM we can work only with two files"  # TODO #3 -- make it work for any number

    # load the files headers
    # see which fields differ
    # call proc_file from ls, with opts.header_fields set to the fields which differ between files

    img1 = nibabel.load(files[0])  # load first image
    img2 = nibabel.load(files[1])  # load second image

    if img1.header.get_data_dtype() != img2.header.get_data_dtype():
        data_dtype = (img1.header.get_data_dtype(), img2.header.get_data_dtype())
    else:
        return "Same data type"

    if img1.header.get_data_shape() != img2.header.get_data_shape():
        data_shape = (img1.header.get_data_shape(), img2.header.get_data_shape())
    else:
        return "Same data shape"

    if img1.header.get_zooms() != img2.header.get_zooms():
        zooms = (img1.header.get_zooms(), img2.header.get_zooms())
    else:
        return "Same voxel sizes"

    # MAIN QUESTION: HOW TO GET 1. properly load files and 2. replace with adjusted header fields?

    opts.header_fields = [data_dtype, data_shape, zooms]  # TODO #1

    from .ls import proc_file
    rows = [proc_file(f, opts) for f in files]

    print(table2string(rows))

    # Later TODO #2
    # if opts.header_fields are specified, then limit comparison only to those
    # fields