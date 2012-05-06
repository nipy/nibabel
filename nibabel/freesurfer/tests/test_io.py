import os
from os.path import join as pjoin

from nose.tools import assert_true
import numpy as np
from numpy.testing import assert_equal

from .. import read_geometry, read_morph_data, read_annot, read_label


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
