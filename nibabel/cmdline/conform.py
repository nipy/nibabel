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
Conform neuroimaging volume to arbitrary shape and voxel size.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

from nibabel import __version__
from nibabel.loadsave import load
from nibabel.processing import conform


def _get_parser():
    """Return command-line argument parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("infile",
                   help="Neuroimaging volume to conform.")
    p.add_argument("outfile",
                   help="Name of output file.")
    p.add_argument("--out-shape", nargs=3, default=(256, 256, 256),
                   help="Shape of the conformed output.")
    p.add_argument("--voxel-size", nargs=3, default=(1, 1, 1),
                   help="Voxel size in millimeters of the conformed output.")
    p.add_argument("--orientation", default="RAS",
                   help="Orientation of the conformed output.")
    p.add_argument("-f", "--force", action="store_true",
                   help="Overwrite existing output files.")
    p.add_argument("-V", "--version", action="version", version="{} {}".format(p.prog, __version__))

    return p


def main(args=None):
    """Main program function."""
    parser = _get_parser()
    if args is None:
        namespace = parser.parse_args(sys.argv[1:])
    else:
        namespace = parser.parse_args(args)

    kwargs = vars(namespace)
    from_img = load(kwargs["infile"])

    if not kwargs["force"] and Path(kwargs["outfile"]).exists():
        raise FileExistsError("Output file exists: {}".format(kwargs["outfile"]))

    out_img = conform(from_img=from_img,
        out_shape=kwargs["out_shape"],
        voxel_size=kwargs["voxel_size"],
        order=3,
        cval=0.0,
        orientation=kwargs["orientation"])

    out_img.to_filename(kwargs["outfile"])
