# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from pdb import set_trace as dbg

from os.path import join as pjoin, dirname

from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises)

DATA_PATH = pjoin(dirname(__file__), 'data')

import nibabel as nib
from nibabel.streamlines.header import Field


def test_load_file():
    # Test loading empty file
    # empty_file = pjoin(DATA_PATH, "empty.trk")
    # empty_trk = nib.streamlines.load(empty_file)

    # hdr = empty_trk.get_header()
    # points = empty_trk.get_points(as_generator=False)
    # scalars = empty_trk.get_scalars(as_generator=False)
    # properties = empty_trk.get_properties(as_generator=False)
    
    # assert_equal(hdr[Field.NB_STREAMLINES], 0)
    # assert_equal(len(points), 0)
    # assert_equal(len(scalars), 0)
    # assert_equal(len(properties), 0)

    # for i in empty_trk: pass  # Check if we can iterate through the streamlines.

    # Test loading non-empty file
    trk_file = pjoin(DATA_PATH, "uncinate.trk")
    trk = nib.streamlines.load(trk_file)

    hdr = trk.get_header()
    points = trk.get_points(as_generator=False)
    1/0
    scalars = trk.get_scalars(as_generator=False)
    properties = trk.get_properties(as_generator=False)
    
    assert_equal(hdr[Field.NB_STREAMLINES] > 0, True)
    assert_equal(len(points) > 0, True)
    #assert_equal(len(scalars), 0)
    #assert_equal(len(properties), 0)

    for i in trk: pass  # Check if we can iterate through the streamlines.
    

if __name__ == "__main__":
    test_load_file()