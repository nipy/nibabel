from __future__ import division, print_function, absolute_import
import os
from os.path import join as pjoin, isdir
import getpass
import time
import hashlib


from ...tmpdirs import InTemporaryDirectory

from nose.tools import assert_true
import numpy as np
from numpy.testing import assert_equal, dec

from .. import (read_geometry, read_morph_data, read_annot, read_label,
                write_geometry, write_annot)

from ...tests.nibabel_data import get_nibabel_data


DATA_SDIR = 'fsaverage'

have_freesurfer = False
if 'SUBJECTS_DIR' in os.environ:
    # May have Freesurfer installed with data
    data_path = pjoin(os.environ["SUBJECTS_DIR"], DATA_SDIR)
    have_freesurfer = isdir(data_path)
else:
    # May have nibabel test data submodule checked out
    nib_data = get_nibabel_data()
    if nib_data != '':
        data_path = pjoin(nib_data, 'nitest-freesurfer', DATA_SDIR)
        have_freesurfer = isdir(data_path)

freesurfer_test = dec.skipif(
    not have_freesurfer,
    'cannot find freesurfer {0} directory'.format(DATA_SDIR))


def _hash_file_content(fname):
    hasher = hashlib.md5()
    with open(fname, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()


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
            read_create_stamp = fobj.readline().decode().rstrip('\n')

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
        hash_ = _hash_file_content(annot_path)

        labels, ctab, names = read_annot(annot_path)
        assert_true(labels.shape == (163842, ))
        assert_true(ctab.shape == (len(names), 5))

        labels_orig = None
        if a == 'aparc':
            labels_orig, _, _ = read_annot(annot_path, orig_ids=True)
            np.testing.assert_array_equal(labels == -1, labels_orig == 0)
            # Handle different version of fsaverage
            if hash_ == 'bf0b488994657435cdddac5f107d21e8':
                assert_true(np.sum(labels_orig == 0) == 13887)
            elif hash_ == 'd4f5b7cbc2ed363ac6fcf89e19353504':
                assert_true(np.sum(labels_orig == 1639705) == 13327)
            else:
                raise RuntimeError("Unknown freesurfer file. Please report "
                                   "the problem to the maintainer of nibabel.")

        # Test equivalence of freesurfer- and nibabel-generated annot files
        # with respect to read_annot()
        with InTemporaryDirectory():
            annot_path = 'test'
            write_annot(annot_path, labels, ctab, names)

            labels2, ctab2, names2 = read_annot(annot_path)
            if labels_orig is not None:
                labels_orig_2, _, _ = read_annot(annot_path, orig_ids=True)

        np.testing.assert_array_equal(labels, labels2)
        if labels_orig is not None:
            np.testing.assert_array_equal(labels_orig, labels_orig_2)
        np.testing.assert_array_equal(ctab, ctab2)
        assert_equal(names, names2)


@freesurfer_test
def test_label():
    """Test IO of .label"""
    label_path = pjoin(data_path, "label", "lh.BA1.label")
    label = read_label(label_path)
    # XXX : test more
    assert_true(np.all(label > 0))

    labels, scalars = read_label(label_path, True)
    assert_true(np.all(labels == label))
    assert_true(len(labels) == len(scalars))
