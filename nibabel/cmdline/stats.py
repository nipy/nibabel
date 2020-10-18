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
Compute image statistics
"""

import argparse
from nibabel.loadsave import load
from nibabel.imagestats import mask_volume


def _get_parser():
    """Return command-line argument parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("infile",
                   help="Neuroimaging volume to compute statistics on.")
    p.add_argument("-V", "--Volume", action="store_true", required=False,
                   help="Compute mask volume of a given mask image.")
    p.add_argument("--units", default="mm3", required=False,
                   help="Preferred output units of {mm3, vox}. Defaults to mm3")
    return p

def main(args=None):
    """Main program function."""
    parser = _get_parser()
    opts = parser.parse_args(args)
    from_img = load(opts.infile)

    if opts.Volume:
        computed_volume = mask_volume(from_img, opts.units)
        print(computed_volume)
        return computed_volume
