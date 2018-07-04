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


def diff_values(first_item, second_item):
    """Generically compares two values, returns true if different"""
    if np.any(first_item != second_item):  # comparing items that are instances of class np.ndarray
        return True

    elif type(first_item) != type(second_item):  # comparing items that differ in data type
        return True

    else:  # all other use cases
        return first_item != second_item


def diff_headers(files, fields):
    """Iterates over all header fields of all files to find those that differ

        Parameters
        ----------
        files: a given list of files to be compared
        fields: the fields to be compared

        Returns
        -------
        list
          header fields whose values differ across files
        """

    headers = []

    for f in range(len(files)):  # for each file
        for h in fields:  # for each header

            # each maneuver is encased in a try block after exceptions have previously occurred
            # get the particular header field within the particular file

            try:
                field = files[f][h]

            except ValueError:
                continue

            # filter numpy arrays with a NaN value
            try:
                if np.all(np.isnan(field)):
                    continue

            except TypeError:
                pass

            # compare current file with other files
            for i in files[f+1:]:
                other_field = i[h]

                # sometimes field.item doesn't work
                try:
                    # converting bytes to be compared as strings
                    if isinstance(field.item(0), bytes):
                        field = field.item(0).decode("utf-8")

                    # converting np.ndarray to lists to remove ambiguity
                    if isinstance(field, np.ndarray):
                        field = field.tolist()

                    if isinstance(other_field.item(0), bytes):
                        other_field = other_field.item(0).decode("utf-8")
                    if isinstance(other_field, np.ndarray):
                        other_field = other_field.tolist()

                except AttributeError:
                    continue

                # if the header values of the two files are different, append
                if diff_values(field, other_field):
                    headers.append(h)

    if headers:  # return a list of headers for the files whose values differ
        return headers


def diff_header_fields(header_field, files):
    """Iterates over a single header field of multiple files

    Parameters
    ----------
    header_field: a given header field
    files: the files to be compared

    Returns
    -------
    list
      str for each value corresponding to each file's given header field
    """

    keyed_inputs = []

    for i in files:

        # each maneuver is encased in a try block after exceptions have previously occurred
        # get the particular header field within the particular file

        try:
            field_value = i[header_field]
        except ValueError:
            continue

        # compare different data types, return all values as soon as diff is found
        for x in files[1:]:
            try:
                data_diff = diff_values(str(x[header_field].dtype), str(field_value.dtype))

                if data_diff:
                    break
            except ValueError:
                continue

        # string formatting of responses
        try:

            # if differences are found among data types
            if data_diff:
                # accounting for how to arrange arrays
                if field_value.ndim < 1:
                    keyed_inputs.append("{}@{}".format(field_value, field_value.dtype))
                elif field_value.ndim == 1:
                    keyed_inputs.append("{}@{}".format(list(field_value), field_value.dtype))

            # if no differences are found among data types
            else:
                if field_value.ndim < 1:
                    keyed_inputs.append(field_value)
                elif field_value.ndim == 1:
                    keyed_inputs.append(list(field_value))

        except UnboundLocalError:
            continue

    for i in range(len(keyed_inputs)):
        keyed_inputs[i] = str(keyed_inputs[i])

    return keyed_inputs


def get_headers_diff(file_headers, headers):
    """Get difference between headers

    Parameters
    ----------
    file_headers: list of actual headers from files
    headers: list of header fields that differ

    Returns
    -------
    dict
      str: list for each header field which differs, return list of
      values per each file
    """
    output = OrderedDict()

    # if there are headers that differ
    if headers:

        # for each header
        for header in headers:

            # find the values corresponding to the files that differ
            val = diff_header_fields(header, file_headers)

            # store these values in a dictionary
            if val:
                output[header] = val

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

    file_headers = [nib.load(f).header for f in files]

    if opts.header_fields:  # will almost always have a header field
        # signals "all fields"
        if opts.header_fields == 'all':
            # TODO: header fields might vary across file types, thus prior sensing would be needed
            header_fields = file_headers[0].keys()
        else:
            header_fields = opts.header_fields.split(',')
    headers = diff_headers(file_headers, header_fields)
    diff = get_headers_diff(file_headers, headers)
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
