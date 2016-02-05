# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
from __future__ import division, print_function, absolute_import

from os.path import join as pjoin

import gzip
import bz2
import warnings
import types
from io import BytesIO

import numpy as np

from .. import load, Nifti1Image
from ..externals.netcdf import netcdf_file
from ..deprecated import ModuleProxy
from .. import minc1
from ..minc1 import Minc1File, Minc1Image, MincHeader

from nose.tools import (assert_true, assert_equal, assert_false, assert_raises)
from numpy.testing import assert_array_equal
from ..tmpdirs import InTemporaryDirectory
from ..testing import data_path

from . import test_spatialimages as tsi
from .test_fileslice import slicer_samples
from .test_helpers import assert_data_similar

EG_FNAME = pjoin(data_path, 'tiny.mnc')

# Example images in format expected for ``test_image_api``, adding ``zooms``
# item.
EXAMPLE_IMAGES = [
    dict(
        fname=pjoin(data_path, 'tiny.mnc'),
        shape=(10, 20, 20),
        dtype=np.uint8,
        affine=np.array([[0, 0, 2.0, -20],
                         [0, 2.0, 0, -20],
                         [2.0, 0, 0, -10],
                         [0, 0, 0, 1]]),
        zooms=(2., 2., 2.),
        # These values from SPM2
        data_summary=dict(
            min=0.20784314,
            max=0.74901961,
            mean=0.60602819),
        is_proxy=True),
    dict(
        fname=pjoin(data_path, 'minc1_1_scale.mnc'),
        shape=(10, 20, 20),
        dtype=np.uint8,
        affine=np.array([[0, 0, 2.0, -20],
                         [0, 2.0, 0, -20],
                         [2.0, 0, 0, -10],
                         [0, 0, 0, 1]]),
        zooms=(2., 2., 2.),
        # These values from mincstats
        data_summary=dict(
            min=0.2082842439,
            max=0.2094327615,
            mean=0.2091292083),
        is_proxy=True),
    dict(
        fname=pjoin(data_path, 'minc1_4d.mnc'),
        shape=(2, 10, 20, 20),
        dtype=np.uint8,
        affine=np.array([[0, 0, 2.0, -20],
                         [0, 2.0, 0, -20],
                         [2.0, 0, 0, -10],
                         [0, 0, 0, 1]]),
        zooms=(1., 2., 2., 2.),
        # These values from mincstats
        data_summary=dict(
            min=0.2078431373,
            max=1.498039216,
            mean=0.9090422837),
        is_proxy=True),
]


def test_old_namespace():
    # Check old names are defined in minc1 module and top level
    # Check warnings raised
    arr = np.arange(24).reshape((2, 3, 4))
    aff = np.diag([2, 3, 4, 1])
    with warnings.catch_warnings(record=True) as warns:
        # Top level import.
        # This import does not trigger an import of the minc.py module, because
        # it's the proxy object.
        from .. import minc
        assert_equal(warns, [])
        # If there was a previous import it will be module, otherwise it will be
        # a proxy
        previous_import = isinstance(minc, types.ModuleType)
        if not previous_import:
            assert_true(isinstance(minc, ModuleProxy))
        old_minc1image = minc.Minc1Image  # just to check it works
        # There may or may not be a warning raised on accessing the proxy,
        # depending on whether the minc.py module is already imported in this
        # test run.
        if not previous_import:
            assert_equal(warns.pop(0).category, FutureWarning)
        from .. import Minc1Image, MincImage
        assert_equal(warns, [])
        # The import from old module is the same as that from new
        assert_true(old_minc1image is Minc1Image)
        # But the old named import, imported from new, is not the same
        assert_false(Minc1Image is MincImage)
        assert_equal(warns, [])
        # Create object using old name
        mimg = MincImage(arr, aff)
        assert_array_equal(mimg.get_data(), arr)
        # Call to create object created warning
        assert_equal(warns.pop(0).category, FutureWarning)
        # Another old name
        from ..minc1 import MincFile, Minc1File
        assert_false(MincFile is Minc1File)
        assert_equal(warns, [])
        mf = MincFile(netcdf_file(EG_FNAME))
        assert_equal(mf.get_data_shape(), (10, 20, 20))
        # Call to create object created warning
        assert_equal(warns.pop(0).category, FutureWarning)


class _TestMincFile(object):
    module = minc1
    file_class = Minc1File
    fname = EG_FNAME
    opener = netcdf_file
    test_files = EXAMPLE_IMAGES

    def test_mincfile(self):
        for tp in self.test_files:
            mnc_obj = self.opener(tp['fname'], 'r')
            mnc = self.file_class(mnc_obj)
            assert_equal(mnc.get_data_dtype().type, tp['dtype'])
            assert_equal(mnc.get_data_shape(), tp['shape'])
            assert_equal(mnc.get_zooms(), tp['zooms'])
            assert_array_equal(mnc.get_affine(), tp['affine'])
            data = mnc.get_scaled_data()
            assert_equal(data.shape, tp['shape'])

    def test_mincfile_slicing(self):
        # Test slicing and scaling of mincfile data
        for tp in self.test_files:
            mnc_obj = self.opener(tp['fname'], 'r')
            mnc = self.file_class(mnc_obj)
            data = mnc.get_scaled_data()
            for slicedef in ((slice(None),),
                             (1,),
                             (slice(None), 1),
                             (1, slice(None)),
                             (slice(None), 1, 1),
                             (1, slice(None), 1),
                             (1, 1, slice(None)),
                             ):
                sliced_data = mnc.get_scaled_data(slicedef)
                assert_array_equal(sliced_data, data[slicedef])

    def test_load(self):
        # Check highest level load of minc works
        for tp in self.test_files:
            img = load(tp['fname'])
            data = img.get_data()
            assert_equal(data.shape, tp['shape'])
            # min, max, mean values from read in SPM2 / minctools
            assert_data_similar(data, tp)
            # check if mnc can be converted to nifti
            ni_img = Nifti1Image.from_image(img)
            assert_array_equal(ni_img.affine, tp['affine'])
            assert_array_equal(ni_img.get_data(), data)

    def test_array_proxy_slicing(self):
        # Test slicing of array proxy
        for tp in self.test_files:
            img = load(tp['fname'])
            arr = img.get_data()
            prox = img.dataobj
            assert_true(prox.is_proxy)
            for sliceobj in slicer_samples(img.shape):
                assert_array_equal(arr[sliceobj], prox[sliceobj])


class TestMinc1File(_TestMincFile):

    def test_compressed(self):
        # we can read minc compressed
        # Not so for MINC2; hence this small sub-class
        for tp in self.test_files:
            content = open(tp['fname'], 'rb').read()
            openers_exts = ((gzip.open, '.gz'), (bz2.BZ2File, '.bz2'))
            with InTemporaryDirectory():
                for opener, ext in openers_exts:
                    fname = 'test.mnc' + ext
                    fobj = opener(fname, 'wb')
                    fobj.write(content)
                    fobj.close()
                    img = self.module.load(fname)
                    data = img.get_data()
                    assert_data_similar(data, tp)
                    del img


# Test the Minc header
def test_header_data_io():
    bio = BytesIO()
    hdr = MincHeader()
    arr = np.arange(24).reshape((2, 3, 4))
    assert_raises(NotImplementedError, hdr.data_to_fileobj, arr, bio)
    assert_raises(NotImplementedError, hdr.data_from_fileobj, bio)


class TestMinc1Image(tsi.TestSpatialImage):
    image_class = Minc1Image
    eg_images = (pjoin(data_path, 'tiny.mnc'),)
    module = minc1

    def test_data_to_from_fileobj(self):
        # Check data_from_fileobj of header raises an error
        for fpath in self.eg_images:
            img = self.module.load(fpath)
            bio = BytesIO()
            arr = np.arange(24).reshape((2, 3, 4))
            assert_raises(NotImplementedError,
                          img.header.data_to_fileobj, arr, bio)
            assert_raises(NotImplementedError,
                          img.header.data_from_fileobj, bio)
