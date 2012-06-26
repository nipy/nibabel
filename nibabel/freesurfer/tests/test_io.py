from __future__ import with_statement
import os
from os.path import join as pjoin
import getpass
import time

from nibabel.tmpdirs import InTemporaryDirectory

from nose.tools import assert_true
import numpy as np
from numpy.testing import assert_equal

from .. import read_geometry, read_morph_data, read_annot, read_label, \
                write_geometry


have_freesurfer = True
if 'SUBJECTS_DIR' not in os.environ:
    # Test suite relies on the definition of SUBJECTS_DIR
    have_freesurfer = False

freesurfer_test = np.testing.dec.skipif(not have_freesurfer,
                                        'SUBJECTS_DIR not set')

if have_freesurfer:
    subj_dir = os.environ["SUBJECTS_DIR"]
    subject_id = 'fsaverage'
    data_path = pjoin(subj_dir, subject_id)


@freesurfer_test
def test_geometry():
    """Test IO of .surf"""
    surf_path = pjoin(data_path, "surf", "%s.%s" % ("lh", "inflated"))
    coords, faces = read_geometry(surf_path)
    assert_equal(0, faces.min())
    assert_equal(coords.shape[0], faces.max() + 1)

    # Test quad with sphere
    surf_path = pjoin(data_path, "surf", "%s.%s" % ("lh", "sphere"))
    coords, faces = read_geometry(surf_path)
    assert_equal(0, faces.min())
    assert_equal(coords.shape[0], faces.max() + 1)

    # Test equivalence of freesurfer- and nibabel-generated triangular files
    # with respect to read_geometry()
    with InTemporaryDirectory():
        surf_path = 'test'
        create_stamp = "created by %s on %s" % (getpass.getuser(),
            time.ctime())
        write_geometry(surf_path, coords, faces, create_stamp)

        coords2, faces2 = read_geometry(surf_path)

        with open(surf_path, 'rb') as fobj:
            magic = np.fromfile(fobj, ">u1", 3)
            read_create_stamp = fobj.readline().rstrip('\n')

    assert_equal(create_stamp, read_create_stamp)

    np.testing.assert_array_equal(coords, coords2)
    np.testing.assert_array_equal(faces, faces2)

    # Validate byte ordering
    coords_swapped = coords.byteswap().newbyteorder()
    faces_swapped = faces.byteswap().newbyteorder()
    np.testing.assert_array_equal(coords_swapped, coords)
    np.testing.assert_array_equal(faces_swapped, faces)


@freesurfer_test
def test_morph_data():
    """Test IO of morphometry data file (eg. curvature)."""
    curv_path = pjoin(data_path, "surf", "%s.%s" % ("lh", "curv"))
    curv = read_morph_data(curv_path)
    assert_true(-1.0 < curv.min() < 0)
    assert_true(0 < curv.max() < 1.0)


@freesurfer_test
def test_annot():
    """Test IO of .annot"""
    annots = ['aparc', 'aparc.a2005s']
    for a in annots:
        annot_path = pjoin(data_path, "label", "%s.%s.annot" % ("lh", a))
        labels, ctab, names = read_annot(annot_path)
        assert_true(labels.shape == (163842, ))
        assert_true(ctab.shape == (len(names), 5))


@freesurfer_test
def test_label():
    """Test IO of .label"""
    label_path = pjoin(data_path, "label", "lh.BA1.label")
    label = read_label(label_path)
    # XXX : test more
    assert_true(np.all(label > 0))
