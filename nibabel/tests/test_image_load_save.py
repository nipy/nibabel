# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Tests for loader function '''
from __future__ import with_statement
import os
from os.path import join as pjoin
import shutil
from tempfile import mkstemp, mkdtemp
from StringIO import StringIO

import numpy as np

# If we don't have scipy, then we cannot write SPM format files
try:
    import scipy.io
except ImportError:
    have_scipy = False
else:
    have_scipy = True


import nibabel as nib
import nibabel.analyze as ana
import nibabel.spm99analyze as spm99
import nibabel.spm2analyze as spm2
import nibabel.nifti1 as ni1
import nibabel.loadsave as nils
from .. import (Nifti1Image, Nifti1Pair, MincImage, Spm2AnalyzeImage,
                Spm99AnalyzeImage, AnalyzeImage)

from ..tmpdirs import InTemporaryDirectory

from ..volumeutils import native_code, swapped_code

from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_true, assert_equal, assert_raises

from ..testing import parametric


def round_trip(img):
    # round trip a nifti single
    sio = StringIO()
    img.file_map['image'].fileobj = sio
    img.to_file_map()
    img2 = nib.Nifti1Image.from_file_map(img.file_map)
    return img2


@parametric
def test_conversion():
    shape = (2, 4, 6)
    affine = np.diag([1, 2, 3, 1])
    for npt in np.float32, np.int16:
        data = np.arange(np.prod(shape), dtype=npt).reshape(shape)
        for r_class_def in nib.class_map.values():
            r_class = r_class_def['class']
            img = r_class(data, affine)
            img.set_data_dtype(npt)
            for w_class_def in nib.class_map.values():
                w_class = w_class_def['class']
                img2 = w_class.from_image(img)
                yield assert_array_equal(img2.get_data(), data)
                yield assert_array_equal(img2.get_affine(), affine)


@parametric
def test_save_load_endian():
    shape = (2, 4, 6)
    affine = np.diag([1, 2, 3, 1])
    data = np.arange(np.prod(shape), dtype='f4').reshape(shape)
    # Native endian image
    img = nib.Nifti1Image(data, affine)
    yield assert_equal(img.get_header().endianness, native_code)
    img2 = round_trip(img)
    yield assert_equal(img2.get_header().endianness, native_code)
    yield assert_array_equal(img2.get_data(), data)
    # byte swapped endian image
    bs_hdr = img.get_header().as_byteswapped()
    bs_img = nib.Nifti1Image(data, affine, bs_hdr)
    yield assert_equal(bs_img.get_header().endianness, swapped_code)
    # of course the data is the same because it's not written to disk
    yield assert_array_equal(bs_img.get_data(), data)
    # Check converting to another image
    cbs_img = nib.AnalyzeImage.from_image(bs_img)
    # this will make the header native by doing the header conversion
    cbs_hdr = cbs_img.get_header()
    yield assert_equal(cbs_hdr.endianness, native_code)
    # and the byte order follows it back into another image
    cbs_img2 = nib.Nifti1Image.from_image(cbs_img)
    cbs_hdr2 = cbs_img2.get_header()
    yield assert_equal(cbs_hdr2.endianness, native_code)
    # Try byteswapped round trip
    bs_img2 = round_trip(bs_img)
    bs_data2 = bs_img2.get_data()
    # now the data dtype was swapped endian, so the read data is too
    yield assert_equal(bs_data2.dtype.byteorder, swapped_code)
    yield assert_equal(bs_img2.get_header().endianness, swapped_code)
    yield assert_array_equal(bs_data2, data)
    # Now mix up byteswapped data and non-byteswapped header
    mixed_img = nib.Nifti1Image(bs_data2, affine)
    yield assert_equal(mixed_img.get_header().endianness, native_code)
    m_img2 = round_trip(mixed_img)
    yield assert_equal(m_img2.get_header().endianness, native_code)
    yield assert_array_equal(m_img2.get_data(), data)
    

@parametric
def test_save_load():
    shape = (2, 4, 6)
    npt = np.float32
    data = np.arange(np.prod(shape), dtype=npt).reshape(shape)
    affine = np.diag([1, 2, 3, 1])
    affine[:3,3] = [3,2,1]
    img = ni1.Nifti1Image(data, affine)
    img.set_data_dtype(npt)
    with InTemporaryDirectory() as pth:
        nifn = 'an_image.nii'
        sifn = 'another_image.img'
        ni1.save(img, nifn)
        re_img = nils.load(nifn)
        yield assert_true(isinstance(re_img, ni1.Nifti1Image))
        yield assert_array_equal(re_img.get_data(), data)
        yield assert_array_equal(re_img.get_affine(), affine)
        # These and subsequent del statements are to prevent confusing
        # windows errors when trying to open files or delete the
        # temporary directory. 
        del re_img
        if have_scipy: # skip we we cannot read .mat files
            spm2.save(img, sifn)
            re_img2 = nils.load(sifn)
            yield assert_true(isinstance(re_img2, spm2.Spm2AnalyzeImage))
            yield assert_array_equal(re_img2.get_data(), data)
            yield assert_array_equal(re_img2.get_affine(), affine)
            del re_img2
            spm99.save(img, sifn)
            re_img3 = nils.load(sifn)
            yield assert_true(isinstance(re_img3,
                                         spm99.Spm99AnalyzeImage))
            yield assert_array_equal(re_img3.get_data(), data)
            yield assert_array_equal(re_img3.get_affine(), affine)
            ni1.save(re_img3, nifn)
            del re_img3
        re_img = nils.load(nifn)
        yield assert_true(isinstance(re_img, ni1.Nifti1Image))
        yield assert_array_equal(re_img.get_data(), data)
        yield assert_array_equal(re_img.get_affine(), affine)
        del re_img


@parametric
def test_two_to_one():
    # test going from two to one file in save
    shape = (2, 4, 6)
    npt = np.float32
    data = np.arange(np.prod(shape), dtype=npt).reshape(shape)
    affine = np.diag([1, 2, 3, 1])
    affine[:3,3] = [3,2,1]
    # single file format
    img = ni1.Nifti1Image(data, affine)
    yield assert_equal(img.get_header()['magic'], 'n+1')
    str_io = StringIO()
    img.file_map['image'].fileobj = str_io
    # check that the single format vox offset is set correctly
    img.to_file_map()
    yield assert_equal(img.get_header()['magic'], 'n+1')
    yield assert_equal(img.get_header()['vox_offset'], 352)
    # make a new pair image, with the single image header
    pimg = ni1.Nifti1Pair(data, affine, img.get_header())
    isio = StringIO()
    hsio = StringIO()
    pimg.file_map['image'].fileobj = isio
    pimg.file_map['header'].fileobj = hsio
    pimg.to_file_map()
    # the offset remains the same
    yield assert_equal(pimg.get_header()['magic'], 'ni1')
    yield assert_equal(pimg.get_header()['vox_offset'], 352)
    yield assert_array_equal(pimg.get_data(), data)
    # same for from_image, going from single image to pair format
    ana_img = ana.AnalyzeImage.from_image(img)
    yield assert_equal(ana_img.get_header()['vox_offset'], 352)
    # back to the single image, save it again to a stringio
    str_io = StringIO()
    img.file_map['image'].fileobj = str_io
    img.to_file_map()
    yield assert_equal(img.get_header()['vox_offset'], 352)
    aimg = ana.AnalyzeImage.from_image(img)
    yield assert_equal(aimg.get_header()['vox_offset'], 352)
    aimg = spm99.Spm99AnalyzeImage.from_image(img)
    yield assert_equal(aimg.get_header()['vox_offset'], 352)
    aimg = spm2.Spm2AnalyzeImage.from_image(img)
    yield assert_equal(aimg.get_header()['vox_offset'], 352)
    nfimg = ni1.Nifti1Pair.from_image(img)
    yield assert_equal(nfimg.get_header()['vox_offset'], 352)
    # now set the vox offset directly
    hdr = nfimg.get_header()
    hdr['vox_offset'] = 0
    yield assert_equal(nfimg.get_header()['vox_offset'], 0)
    # check it gets properly set by the nifti single image
    nfimg = ni1.Nifti1Image.from_image(img)
    yield assert_equal(nfimg.get_header()['vox_offset'], 352)
    
    
@parametric
def test_negative_load_save():
    shape = (1,2,5)
    data = np.arange(10).reshape(shape) - 10.0
    affine = np.eye(4)
    hdr = nib.Nifti1Header()
    hdr.set_data_dtype(np.int16)
    img = nib.Nifti1Image(data, affine, hdr)
    str_io = StringIO()
    img.file_map['image'].fileobj = str_io
    img.to_file_map()
    str_io.seek(0)
    re_img = nib.Nifti1Image.from_file_map(img.file_map)
    yield assert_array_almost_equal(re_img.get_data(), data, 4)


def test_filename_save():
    # This is to test the logic in the load and save routines, relating
    # extensions to filetypes
    # Tuples of class, ext, loadedclass
    inklass_ext_loadklasses = (
        (Nifti1Image, '.nii', Nifti1Image),
        (Nifti1Image, '.img', Nifti1Pair),
        (MincImage, '.nii', Nifti1Image),
        (MincImage, '.img', Nifti1Pair),
        (Spm2AnalyzeImage, '.nii', Nifti1Image),
        (Spm2AnalyzeImage, '.img', Spm2AnalyzeImage),
        (Spm99AnalyzeImage, '.nii', Nifti1Image),
        (Spm99AnalyzeImage, '.img', Spm2AnalyzeImage),
        (AnalyzeImage, '.nii', Nifti1Image),
        (AnalyzeImage, '.img', Spm2AnalyzeImage),
    )
    shape = (2, 4, 6)
    affine = np.diag([1, 2, 3, 1])
    data = np.arange(np.prod(shape), dtype='f4').reshape(shape)
    for inklass, out_ext, loadklass in inklass_ext_loadklasses:
        if not have_scipy:
            # We can't load a SPM analyze type without scipy.  These types have
            # a 'mat' file (the type we can't load)
            if ('mat', '.mat') in loadklass.files_types:
                continue
        img = inklass(data, affine)
        try:
            pth = mkdtemp()
            fname = pjoin(pth, 'image' + out_ext)
            nib.save(img, fname)
            rt_img = nib.load(fname)
            assert_array_almost_equal(rt_img.get_data(), data)
            assert_true(type(rt_img) is loadklass)
            # delete image to allow file close.  Otherwise windows
            # raises an error when trying to delete the directory
            del rt_img
        finally:
            shutil.rmtree(pth)
