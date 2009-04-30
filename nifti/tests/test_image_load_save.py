''' Tests for loader function '''

import os
import tempfile

from StringIO import StringIO

import numpy as np

import nifti as nf
import nifti.analyze as ana
import nifti.spm99analyze as spm99
import nifti.spm2analyze as spm2
import nifti.nifti1 as ni1
import nifti.loadsave as nils

from nifti.volumeutils import native_code, swapped_code

from numpy.testing import assert_array_equal
from nose.tools import assert_true, assert_equal, assert_raises


def test_conversion():
    shape = (2, 4, 6)
    affine = np.diag([1, 2, 3, 1])
    for npt in np.float32, np.int16:
        data = np.arange(np.prod(shape), dtype=npt).reshape(shape)
        img = ni1.Nifti1Image(data, affine)
        img.set_data_dtype(npt)
        img2 = spm2.Spm2AnalyzeImage.from_image(img)
        yield assert_array_equal, img2.get_data(), data
        img3 = spm99.Spm99AnalyzeImage.from_image(img)
        yield assert_array_equal, img3.get_data(), data
        img4 = ana.AnalyzeImage.from_image(img)
        yield assert_array_equal, img4.get_data(), data
        img5 = ni1.Nifti1Image.from_image(img4)
        yield assert_array_equal, img5.get_data(), data


def test_endianness():
    shape = (2, 4, 6)
    affine = np.diag([1, 2, 3, 1])
    data = np.ones(shape)
    img = nf.Nifti1Image(data, affine)
    hdr = img.get_header()
    yield assert_equal, hdr.endianness, native_code
    bs_hdr = hdr.as_byteswapped()
    img2 = nf.Nifti1Image(data, affine, bs_hdr)
    hdr2 = img2.get_header()
    yield assert_equal, hdr2.endianness, swapped_code
    img3 = nf.AnalyzeImage.from_image(img2)
    hdr3 = img3.get_header()
    yield assert_equal, hdr3.endianness, swapped_code
    img4 = nf.Nifti1Image.from_image(img3)
    hdr4 = img4.get_header()
    yield assert_equal, hdr3.endianness, swapped_code    

def test_save_load():
    shape = (2, 4, 6)
    npt = np.float32
    data = np.arange(np.prod(shape), dtype=npt).reshape(shape)
    affine = np.diag([1, 2, 3, 1])
    affine[:3,3] = [3,2,1]
    img = ni1.Nifti1Image(data, affine)
    img.set_data_dtype(npt)
    try:
        _, nifn = tempfile.mkstemp('.nii')
        # this somewhat unsafe, because we will make .hdr and .mat files too
        _, sifn = tempfile.mkstemp('.img')
        ni1.save(img, nifn)
        re_img = nils.load(nifn)
        yield assert_true, isinstance(re_img, ni1.Nifti1Image)
        yield assert_array_equal, re_img.get_data(), data
        yield assert_array_equal, re_img.get_affine(), affine
        spm2.save(img, sifn)
        re_img2 = nils.load(sifn)
        yield assert_true, isinstance(re_img2, spm2.Spm2AnalyzeImage)
        yield assert_array_equal, re_img2.get_data(), data
        yield assert_array_equal, re_img2.get_affine(), affine
        spm99.save(img, sifn)
        re_img3 = nils.load(sifn)
        yield assert_true, isinstance(re_img3, spm99.Spm99AnalyzeImage)
        yield assert_array_equal, re_img3.get_data(), data
        yield assert_array_equal, re_img3.get_affine(), affine
        ni1.save(re_img3, nifn)
        re_img = nils.load(nifn)
        yield assert_true, isinstance(re_img, ni1.Nifti1Image)
        yield assert_array_equal, re_img.get_data(), data
        yield assert_array_equal, re_img.get_affine(), affine
    finally:
        os.unlink(nifn)
        os.unlink(sifn)
        os.unlink(sifn[:-4] + '.hdr')
        os.unlink(sifn[:-4] + '.mat')


def test_two_to_one():
    # test going from two to one file in save
    shape = (2, 4, 6)
    npt = np.float32
    data = np.arange(np.prod(shape), dtype=npt).reshape(shape)
    affine = np.diag([1, 2, 3, 1])
    affine[:3,3] = [3,2,1]
    img = ni1.Nifti1Image(data, affine)
    yield assert_equal, img.get_header()['magic'], 'n+1'
    str_io = StringIO()
    files = {'header':str_io, 'image':str_io}
    img.to_files(files)
    yield assert_equal, img.get_header()['magic'], 'n+1'
    yield assert_equal, img.get_header()['vox_offset'], 352
    str_io2 = StringIO()
    files['image'] = str_io2
    img.to_files(files)
    yield assert_equal, img.get_header()['magic'], 'ni1'
    yield assert_equal, img.get_header()['vox_offset'], 0
