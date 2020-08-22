""" Testing loadsave module
"""

from os.path import dirname, join as pjoin
import shutil
import pathlib

import numpy as np

from .. import (Spm99AnalyzeImage, Spm2AnalyzeImage,
                Nifti1Pair, Nifti1Image,
                Nifti2Pair, Nifti2Image)
from ..loadsave import load, read_img_data
from ..filebasedimages import ImageFileError
from ..tmpdirs import InTemporaryDirectory, TemporaryDirectory

from ..optpkg import optional_package
_, have_scipy, _ = optional_package('scipy')

from numpy.testing import (assert_almost_equal,
                           assert_array_equal)

import pytest

data_path = pjoin(dirname(__file__), 'data')


def test_read_img_data():
    fnames_test = [
        'example4d.nii.gz',
        'example_nifti2.nii.gz',
        'minc1_1_scale.mnc',
        'minc1_4d.mnc',
        'test.mgz',
        'tiny.mnc'
    ]
    fnames_test += [pathlib.Path(p) for p in fnames_test]
    for fname in fnames_test:
        # os.path.join doesnt work between str / os.PathLike in py3.5
        fpath = pjoin(data_path, str(fname))
        if isinstance(fname, pathlib.Path):
            fpath = pathlib.Path(fpath)
        img = load(fpath)
        data = img.get_fdata()
        with pytest.deprecated_call():
            data2 = read_img_data(img)
        assert_array_equal(data, data2)
        # These examples have null scaling - assert prefer=unscaled is the same
        dao = img.dataobj
        if hasattr(dao, 'slope') and hasattr(img.header, 'raw_data_from_fileobj'):
            assert (dao.slope, dao.inter) == (1, 0)
            with pytest.deprecated_call():
                assert_array_equal(read_img_data(img, prefer='unscaled'), data)
        # Assert all caps filename works as well
        with TemporaryDirectory() as tmpdir:
            up_fpath = pjoin(tmpdir, str(fname).upper())
            if isinstance(fname, pathlib.Path):
                up_fpath = pathlib.Path(up_fpath)
            # shutil doesnt work with os.PathLike in py3.5
            shutil.copyfile(str(fpath), str(up_fpath))
            img = load(up_fpath)
            assert_array_equal(img.dataobj, data)
            del img


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load('does_not_exist.nii.gz')


def test_load_empty_image():
    with InTemporaryDirectory():
        open('empty.nii', 'w').close()
        with pytest.raises(ImageFileError) as err:
            load('empty.nii')
    assert str(err.value).startswith('Empty file: ')


def test_read_img_data_nifti():
    shape = (2, 3, 4)
    data = np.random.normal(size=shape)
    out_dtype = np.dtype(np.int16)
    classes = (Nifti1Pair, Nifti1Image, Nifti2Pair, Nifti2Image)
    if have_scipy:
        classes += (Spm99AnalyzeImage, Spm2AnalyzeImage)
    with InTemporaryDirectory():
        for i, img_class in enumerate(classes):
            img = img_class(data, np.eye(4))
            img.set_data_dtype(out_dtype)
            # No filemap => error
            with pytest.deprecated_call(), pytest.raises(ImageFileError):
                read_img_data(img)
            # Make a filemap
            froot = f'an_image_{i}'
            img.file_map = img.filespec_to_file_map(froot)
            # Trying to read from this filemap will generate an error because
            # we are going to read from files that do not exist
            with pytest.deprecated_call(), pytest.raises(IOError):
                read_img_data(img)
            img.to_file_map()
            # Load - now the scaling and offset correctly applied
            img_fname = img.file_map['image'].filename
            img_back = load(img_fname)
            data_back = img_back.get_fdata()
            with pytest.deprecated_call():
                assert_array_equal(data_back, read_img_data(img_back))
            # This is the same as if we loaded the image and header separately
            hdr_fname = (img.file_map['header'].filename
                         if 'header' in img.file_map else img_fname)
            with open(hdr_fname, 'rb') as fobj:
                hdr_back = img_back.header_class.from_fileobj(fobj)
            with open(img_fname, 'rb') as fobj:
                scaled_back = hdr_back.data_from_fileobj(fobj)
            assert_array_equal(data_back, scaled_back)
            # Unscaled is the same as returned from raw_data_from_fileobj
            with open(img_fname, 'rb') as fobj:
                unscaled_back = hdr_back.raw_data_from_fileobj(fobj)
            with pytest.deprecated_call():
                assert_array_equal(unscaled_back, read_img_data(img_back, prefer='unscaled'))
            # If we futz with the scaling in the header, the result changes
            with pytest.deprecated_call():
                assert_array_equal(data_back, read_img_data(img_back))
            has_inter = hdr_back.has_data_intercept
            old_slope = hdr_back['scl_slope']
            old_inter = hdr_back['scl_inter'] if has_inter else 0
            est_unscaled = (data_back - old_inter) / old_slope
            with pytest.deprecated_call():
                actual_unscaled = read_img_data(img_back, prefer='unscaled')
            assert_almost_equal(est_unscaled, actual_unscaled)
            img_back.header['scl_slope'] = 2.1
            if has_inter:
                new_inter = 3.14
                img_back.header['scl_inter'] = 3.14
            else:
                new_inter = 0
            # scaled scaling comes from new parameters in header
            with pytest.deprecated_call():
                assert np.allclose(actual_unscaled * 2.1 + new_inter,
                                   read_img_data(img_back))
            # Unscaled array didn't change
            with pytest.deprecated_call():
                assert_array_equal(actual_unscaled,
                                   read_img_data(img_back, prefer='unscaled'))
            # Check the offset too
            img.header.set_data_offset(1024)
            # Delete arrays still pointing to file, so Windows can re-use
            del actual_unscaled, unscaled_back
            img.to_file_map()
            # Write an integer of zeros after
            with open(img_fname, 'ab') as fobj:
                fobj.write(b'\x00\x00')
            img_back = load(img_fname)
            data_back = img_back.get_fdata()
            with pytest.deprecated_call():
                assert_array_equal(data_back, read_img_data(img_back))
            img_back.header.set_data_offset(1026)
            # Check we pick up new offset
            exp_offset = np.zeros((data.size,), data.dtype) + old_inter
            exp_offset[:-1] = np.ravel(data_back, order='F')[1:]
            exp_offset = np.reshape(exp_offset, shape, order='F')
            with pytest.deprecated_call():
                assert_array_equal(exp_offset, read_img_data(img_back))
            # Delete stuff that might hold onto file references
            del img, img_back, data_back
