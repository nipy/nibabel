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

        Option("--ma", "--data-max-abs-diff",
               dest="data_max_abs_diff",
               type=float,
               default=0.0,
               help="Maximal absolute difference in data between files to tolerate."),

        Option("--mr", "--data-max-rel-diff",
               dest="data_max_rel_diff",
               type=float,
               default=0.0,
               help="Maximal relative difference in data between files to tolerate."
                    " If --data-max-abs-diff is also specified, only the data points "
                    " with absolute difference greater than that value would be "
                    " considered for relative difference check."),
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
        elif isinstance(value0, np.ndarray):
            return np.any(value0 != value)

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


def get_data_hash_diff(files):
    """Get difference between md5 values of data

        Parameters
        ----------
        files: list of actual files

        Returns
        -------
        list
          np.array: md5 values of respective files
        """

    md5sums = [
        hashlib.md5(np.ascontiguousarray(nib.load(f).get_fdata())).hexdigest()
        for f in files
    ]

    if len(set(md5sums)) == 1:
        return []

    return md5sums


def get_data_diff(files, max_abs=0, max_rel=0):
    """Get difference between data

    Parameters
    ----------
    files: list of (str or ndarray)
      If list of strings is provided -- they must be existing file names
    max_abs: float, optional
      Maximal absolute difference to tolerate.
    max_rel: float, optional
      Maximal relative (`abs(diff)/mean(diff)`) difference to tolerate.
      If `max_abs` is specified, then those data points with lesser than that
      absolute difference, are not considered for relative difference testing

    Returns
    -------
    diffs: OrderedDict
        An ordered dict with a record per each file which has differences
        with other files subsequent detected. Each record is a list of
        difference records, one per each file pair.
        Each difference record is an Ordered Dict with possible keys
        'abs' or 'rel' showing maximal absolute or relative differences
        in the file or the record ('CMP': 'incompat') if file shapes
        are incompatible.
    """

    # we are doomed to keep them in RAM now
    data = [f if isinstance(f, np.ndarray) else nib.load(f).get_fdata()
            for f in files]
    diffs = OrderedDict()
    for i, d1 in enumerate(data[:-1]):
        # populate empty entries for non-compared
        diffs1 = [None] * (i + 1)

        for j, d2 in enumerate(data[i + 1:], i + 1):

            if d1.shape == d2.shape:
                abs_diff = np.abs(d1 - d2)
                mean_abs = (np.abs(d1) + np.abs(d2)) * 0.5
                candidates = np.logical_or(mean_abs != 0, abs_diff != 0)

                if max_abs:
                    candidates[abs_diff <= max_abs] = False

                max_abs_diff = np.max(abs_diff)
                if np.any(candidates):
                    rel_diff = abs_diff[candidates] / mean_abs[candidates]
                    if max_rel:
                        sub_thr = rel_diff <= max_rel
                        # Since we operated on sub-selected values already, we need
                        # to plug them back in
                        candidates[
                            tuple((indexes[sub_thr] for indexes in np.where(candidates)))
                        ] = False
                    max_rel_diff = np.max(rel_diff)
                else:
                    max_rel_diff = 0

                if np.any(candidates):

                    diff_rec = OrderedDict()  # so that abs goes before relative

                    diff_rec['abs'] = max_abs_diff
                    diff_rec['rel'] = max_rel_diff
                    diffs1.append(diff_rec)
                else:
                    diffs1.append(None)

            else:
                diffs1.append({'CMP': "incompat"})

        if any(diffs1):

            diffs['DATA(diff %d:)' % (i + 1)] = diffs1

    return diffs


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
    filename_width = "{:<53}"
    value_width = "{:<55}"

    output += "These files are different.\n"
    output += field_width.format('Field/File')

    for i, f in enumerate(files, 1):
        output += "%d:%s" % (i, filename_width.format(os.path.basename(f)))

    output += "\n"

    for key, value in diff.items():
        output += field_width.format(key)

        for item in value:
            if isinstance(item, dict):
                item_str = ', '.join('%s: %s' % i for i in item.items())
            elif item is None:
                item_str = '-'
            else:
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


def diff(files, header_fields='all', data_max_abs_diff=None, data_max_rel_diff=None):
    assert len(files) >= 2, "Please enter at least two files"

    file_headers = [nib.load(f).header for f in files]

    # signals "all fields"
    if header_fields == 'all':
        # TODO: header fields might vary across file types, thus prior sensing would be needed
        header_fields = file_headers[0].keys()
    else:
        header_fields = header_fields.split(',')

    diff = get_headers_diff(file_headers, header_fields)

    data_md5_diffs = get_data_hash_diff(files)
    if data_md5_diffs:
        # provide details, possibly triggering the ignore of the difference
        # in data
        data_diffs = get_data_diff(files,
                                   max_abs=data_max_abs_diff,
                                   max_rel=data_max_rel_diff)
        if data_diffs:
            diff['DATA(md5)'] = data_md5_diffs
            diff.update(data_diffs)

    return diff


def main(args=None, out=None):
    """Getting the show on the road"""

    out = out or sys.stdout
    parser = get_opt_parser()
    (opts, files) = parser.parse_args(args)

    nibabel.cmdline.utils.verbose_level = opts.verbose

    if nibabel.cmdline.utils.verbose_level < 3:
        # suppress nibabel format-compliance warnings
        nib.imageglobals.logger.level = 50

    files_diff = diff(
        files,
        header_fields=opts.header_fields,
        data_max_abs_diff=opts.data_max_abs_diff,
        data_max_rel_diff=opts.data_max_rel_diff
    )

    if files_diff:
        out.write(display_diff(files, files_diff))
        raise SystemExit(1)
    else:
        out.write("These files are identical.\n")
        raise SystemExit(0)
