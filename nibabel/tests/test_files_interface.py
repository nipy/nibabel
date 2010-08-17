# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Testing filesets - a draft

"""
import os
from tempfile import mkstemp

from cStringIO import StringIO

import numpy as np

import nibabel as nib
from ..fileholders import FileHolderError

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from ..testing import parametric


@parametric
def test_files_images():
    # test files creation in image classes
    arr = np.zeros((2,3,4))
    aff = np.eye(4)
    for img_def in nib.class_map.values():
        klass = img_def['class']
        file_map = klass.make_file_map()
        for key, value in file_map.items():
            yield assert_equal(value.filename, None)
            yield assert_equal(value.fileobj, None)
            yield assert_equal(value.pos, 0)
        img = klass(arr, aff)
        for key, value in img.file_map.items():
            yield assert_equal(value.filename, None)
            yield assert_equal(value.fileobj, None)
            yield assert_equal(value.pos, 0)


@parametric
def test_files_interface():
    # test high-level interface to files mapping
    arr = np.zeros((2,3,4))
    aff = np.eye(4)
    img = nib.Nifti1Image(arr, aff)
    # single image
    img.set_filename('test')
    yield assert_equal(img.get_filename(), 'test.nii')
    yield assert_equal(img.file_map['image'].filename, 'test.nii')
    yield assert_raises(KeyError, img.file_map.__getitem__, 'header')
    # pair - note new class
    img = nib.Nifti1Pair(arr, aff)
    img.set_filename('test')
    yield assert_equal(img.get_filename(), 'test.img')
    yield assert_equal(img.file_map['image'].filename, 'test.img')
    yield assert_equal(img.file_map['header'].filename, 'test.hdr')
    # fileobjs - single image
    img = nib.Nifti1Image(arr, aff)
    img.file_map['image'].fileobj = StringIO()
    img.to_file_map() # saves to files
    img2 = nib.Nifti1Image.from_file_map(img.file_map)
    # img still has correct data
    yield assert_array_equal(img2.get_data(), img.get_data())
    # fileobjs - pair
    img = nib.Nifti1Pair(arr, aff)
    img.file_map['image'].fileobj = StringIO()
    # no header yet
    yield assert_raises(FileHolderError, img.to_file_map)
    img.file_map['header'].fileobj = StringIO()
    img.to_file_map() # saves to files
    img2 = nib.Nifti1Pair.from_file_map(img.file_map)
    # img still has correct data
    yield assert_array_equal(img2.get_data(), img.get_data())
    

@parametric
def test_round_trip():
   # write an image to files
   from StringIO import StringIO
   data = np.arange(24, dtype='i4').reshape((2,3,4))
   aff = np.eye(4)
   klasses = [val['class'] for key, val in nib.class_map.items()
              if val['rw']]
   for klass in klasses:
       file_map = klass.make_file_map()
       for key in file_map:
           file_map[key].fileobj = StringIO()
       img = klass(data, aff)
       img.file_map = file_map
       img.to_file_map()
       # read it back again from the written files
       img2 = klass.from_file_map(file_map)
       yield assert_array_equal(img2.get_data(), data)
       # write, read it again
       img2.to_file_map()
       img3 = klass.from_file_map(file_map)
       yield assert_array_equal(img3.get_data(), data)
