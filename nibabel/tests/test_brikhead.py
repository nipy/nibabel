# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from os.path import join as pjoin

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from .. import Nifti1Image, brikhead
from ..testing import assert_data_similar, data_path
from .test_fileslice import slicer_samples

EXAMPLE_IMAGES = [
    dict(
        head=pjoin(data_path, 'example4d+orig.HEAD'),
        fname=pjoin(data_path, 'example4d+orig.BRIK.gz'),
        shape=(33, 41, 25, 3),
        dtype=np.int16,
        affine=np.array(
            [
                [-3.0, 0, 0, 49.5],
                [0, -3.0, 0, 82.312],
                [0, 0, 3.0, -52.3511],
                [0, 0, 0, 1.0],
            ]
        ),
        zooms=(3.0, 3.0, 3.0, 3.0),
        data_summary=dict(min=0, max=13722, mean=4266.76024636),
        is_proxy=True,
        space='ORIG',
        labels=['#0', '#1', '#2'],
        scaling=None,
    ),
    dict(
        head=pjoin(data_path, 'scaled+tlrc.HEAD'),
        fname=pjoin(data_path, 'scaled+tlrc.BRIK'),
        shape=(47, 54, 43, 1.0),
        dtype=np.int16,
        affine=np.array(
            [
                [3.0, 0, 0, -66.0],
                [0, 3.0, 0, -87.0],
                [0, 0, 3.0, -54.0],
                [0, 0, 0, 1.0],
            ]
        ),
        zooms=(3.0, 3.0, 3.0, 0.0),
        data_summary=dict(
            min=1.9416814999999998e-07, max=0.0012724615542099998, mean=0.00023919645351876782
        ),
        is_proxy=True,
        space='TLRC',
        labels=['#0'],
        scaling=np.array([3.88336300e-08]),
    ),
]

EXAMPLE_BAD_IMAGES = [
    dict(head=pjoin(data_path, 'bad_datatype+orig.HEAD'), err=brikhead.AFNIImageError),
    dict(head=pjoin(data_path, 'bad_attribute+orig.HEAD'), err=brikhead.AFNIHeaderError),
]


class TestAFNIHeader:
    module = brikhead
    test_files = EXAMPLE_IMAGES

    def test_makehead(self):
        for tp in self.test_files:
            head1 = self.module.AFNIHeader.from_fileobj(tp['head'])
            head2 = self.module.AFNIHeader.from_header(head1)
            assert head1 == head2
            with pytest.raises(self.module.AFNIHeaderError):
                self.module.AFNIHeader.from_header(header=None)
            with pytest.raises(self.module.AFNIHeaderError):
                self.module.AFNIHeader.from_header(tp['fname'])


class TestAFNIImage:
    module = brikhead
    test_files = EXAMPLE_IMAGES

    def test_brikheadfile(self):
        for tp in self.test_files:
            brik = self.module.load(tp['fname'])
            assert brik.get_data_dtype().type == tp['dtype']
            assert brik.shape == tp['shape']
            assert brik.header.get_zooms() == tp['zooms']
            assert_array_equal(brik.affine, tp['affine'])
            assert brik.header.get_space() == tp['space']
            data = brik.get_fdata()
            assert data.shape == tp['shape']
            assert_array_equal(brik.dataobj.scaling, tp['scaling'])
            assert brik.header.get_volume_labels() == tp['labels']

    def test_load(self):
        # Check highest level load of brikhead works
        for tp in self.test_files:
            img = self.module.load(tp['head'])
            data = img.get_fdata()
            assert data.shape == tp['shape']
            # min, max, mean values
            assert_data_similar(data, tp)
            # check if file can be converted to nifti
            ni_img = Nifti1Image.from_image(img)
            assert_array_equal(ni_img.affine, tp['affine'])
            assert_array_equal(ni_img.get_fdata(), data)

    def test_array_proxy_slicing(self):
        # Test slicing of array proxy
        for tp in self.test_files:
            img = self.module.load(tp['fname'])
            arr = img.get_fdata()
            prox = img.dataobj
            assert prox.is_proxy
            for sliceobj in slicer_samples(img.shape):
                assert_array_equal(arr[sliceobj], prox[sliceobj])


class TestBadFiles:
    module = brikhead
    test_files = EXAMPLE_BAD_IMAGES

    def test_brikheadfile(self):
        for tp in self.test_files:
            with pytest.raises(tp['err']):
                self.module.load(tp['head'])


class TestBadVars:
    module = brikhead
    vars = [
        'type = badtype-attribute\nname = BRICK_TYPES\ncount = 1\n1\n',
        'type = integer-attribute\ncount = 1\n1\n',
    ]

    def test_unpack_var(self):
        for var in self.vars:
            with pytest.raises(self.module.AFNIHeaderError):
                self.module._unpack_var(var)


def test_complex_brick(tmp_path):
    # BRICK_TYPES 5 is complex64: two float32 values per voxel
    info = brikhead.parse_AFNI_header(EXAMPLE_IMAGES[0]['head'])
    info.update(
        DATASET_DIMENSIONS=[4, 5, 6],
        DATASET_RANK=[3, 1, 0, 0, 0, 0, 0, 0],
        BRICK_TYPES=5,
        BRICK_FLOAT_FACS=0,
    )
    hdr = brikhead.AFNIHeader(info)
    assert hdr.get_data_dtype() == np.dtype('<c8')
    data = (
        (np.arange(120) + 1j * np.arange(120, 0, -1))
        .astype('<c8')
        .reshape(hdr.get_data_shape(), order='F')
    )
    brik = tmp_path / 'complex+orig.BRIK'
    brik.write_bytes(data.tobytes(order='F'))
    assert_array_equal(np.asanyarray(brikhead.AFNIArrayProxy(str(brik), hdr)), data)


@pytest.mark.parametrize('tp', EXAMPLE_IMAGES)
def test_optional_attributes(tp, tmp_path):
    # BYTEORDER_STRING and IJK_TO_DICOM_REAL are optional; without them, the
    # byte order is native and the affine comes from ORIGIN, DELTA and
    # ORIENT_SPECIFIC
    with open(tp['head']) as fobj:
        attrs = fobj.read().split('\n\n')
    optional = {'BYTEORDER_STRING', 'IJK_TO_DICOM_REAL'}
    attrs = [attr for attr in attrs if optional.isdisjoint(brikhead.NAME_RE.findall(attr))]
    head = tmp_path / 'minimal+orig.HEAD'
    head.write_text('\n\n'.join(attrs))
    img = brikhead.load(str(head))
    assert optional.isdisjoint(img.header.info)
    assert img.get_data_dtype() == np.dtype(tp['dtype'])  # native
    assert_array_equal(img.affine, tp['affine'])


def test_affine_from_cardinal_axes():
    info = brikhead.parse_AFNI_header(EXAMPLE_IMAGES[0]['head'])
    del info['IJK_TO_DICOM_REAL']
    # Dataset axes are A-P, S-I, L-R
    info.update(ORIENT_SPECIFIC=[3, 5, 1], ORIGIN=[-10.0, 20.0, 30.0], DELTA=[2.0, -3.0, -4.0])
    affine = brikhead.AFNIHeader(info).get_affine()
    # Center of voxel (i, j, k) is at dataset coordinates ORIGIN + ijk * DELTA,
    # which are (y, z, x) in DICOM (LPS) coordinates
    for ijk in [(0, 0, 0), (1, 2, 3), (5, 0, 7)]:
        dset_xyz = np.array(info['ORIGIN']) + np.array(ijk) * info['DELTA']
        lps = dset_xyz[[2, 0, 1]]
        ras = lps * [-1, -1, 1]
        assert_array_equal(affine @ (*ijk, 1), (*ras, 1))
