# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test scripts

Test running scripts
"""
from __future__ import division, print_function, absolute_import

import os
from os.path import (dirname, join as pjoin, abspath, splitext, basename,
                     exists)
import re
from glob import glob

import numpy as np

from ..tmpdirs import InTemporaryDirectory
from ..loadsave import load
from ..orientations import flip_axis, aff2axcodes, inv_ornt_aff

from nose.tools import (assert_true, assert_false, assert_not_equal,
                        assert_equal)

from numpy.testing import assert_almost_equal, assert_array_equal

from .scriptrunner import ScriptRunner
from .nibabel_data import needs_nibabel_data
from ..testing import assert_dt_equal
from .test_parrec import (DTI_PAR_BVECS, DTI_PAR_BVALS,
                          EXAMPLE_IMAGES as PARREC_EXAMPLES)
from .test_parrec_data import BALLS, AFF_OFF
from .test_helpers import assert_data_similar


def _proc_stdout(stdout):
    stdout_str = stdout.decode('latin1').strip()
    return stdout_str.replace(os.linesep, '\n')


runner = ScriptRunner(
    script_sdir = 'bin',
    debug_print_var = 'NIPY_DEBUG_PRINT',
    output_processor=_proc_stdout)
run_command = runner.run_command


def script_test(func):
    # Decorator to label test as a script_test
    func.script_test = True
    return func
script_test.__test__ = False # It's not a test

DATA_PATH = abspath(pjoin(dirname(__file__), 'data'))


@script_test
def test_nib_ls():
    # test nib-ls script
    fname = pjoin(DATA_PATH, 'example4d.nii.gz')
    expected_re = (" (int16|[<>]i2) \[128,  96,  24,   2\] "
                   "2.00x2.00x2.20x2000.00  #exts: 2 sform$")
    cmd = ['nib-ls', fname]
    code, stdout, stderr = run_command(cmd)
    assert_equal(fname, stdout[:len(fname)])
    assert_not_equal(re.match(expected_re, stdout[len(fname):]), None)


@script_test
def test_nib_nifti_dx():
    # Test nib-nifti-dx script
    clean_hdr = pjoin(DATA_PATH, 'nifti1.hdr')
    cmd = ['nib-nifti-dx', clean_hdr]
    code, stdout, stderr = run_command(cmd)
    assert_equal(stdout.strip(), 'Header for "%s" is clean' % clean_hdr)
    dirty_hdr = pjoin(DATA_PATH, 'analyze.hdr')
    cmd = ['nib-nifti-dx', dirty_hdr]
    code, stdout, stderr = run_command(cmd)
    expected = """Picky header check output for "%s"

pixdim[0] (qfac) should be 1 (default) or -1
magic string "" is not valid
sform_code 11776 not valid""" % (dirty_hdr,)
    # Split strings to remove line endings
    assert_equal(stdout, expected)


def vox_size(affine):
    return np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))


def check_conversion(cmd, pr_data, out_fname):
    run_command(cmd)
    img = load(out_fname)
    # Check orientations always LAS
    assert_equal(aff2axcodes(img.affine), tuple('LAS'))
    data = img.get_data()
    assert_true(np.allclose(data, pr_data))
    assert_true(np.allclose(img.header['cal_min'], data.min()))
    assert_true(np.allclose(img.header['cal_max'], data.max()))
    del img, data  # for windows to be able to later delete the file
    # Check minmax options
    run_command(cmd + ['--minmax', '1', '2'])
    img = load(out_fname)
    data = img.get_data()
    assert_true(np.allclose(data, pr_data))
    assert_true(np.allclose(img.header['cal_min'], 1))
    assert_true(np.allclose(img.header['cal_max'], 2))
    del img, data  # for windows
    run_command(cmd + ['--minmax', 'parse', '2'])
    img = load(out_fname)
    data = img.get_data()
    assert_true(np.allclose(data, pr_data))
    assert_true(np.allclose(img.header['cal_min'], data.min()))
    assert_true(np.allclose(img.header['cal_max'], 2))
    del img, data  # for windows
    run_command(cmd + ['--minmax', '1', 'parse'])
    img = load(out_fname)
    data = img.get_data()
    assert_true(np.allclose(data, pr_data))
    assert_true(np.allclose(img.header['cal_min'], 1))
    assert_true(np.allclose(img.header['cal_max'], data.max()))
    del img, data


@script_test
def test_parrec2nii():
    # Test parrec2nii script
    cmd = ['parrec2nii', '--help']
    code, stdout, stderr = run_command(cmd)
    assert_true(stdout.startswith('Usage'))
    with InTemporaryDirectory():
        for eg_dict in PARREC_EXAMPLES:
            fname = eg_dict['fname']
            run_command(['parrec2nii', fname])
            out_froot = splitext(basename(fname))[0] + '.nii'
            img = load(out_froot)
            assert_equal(img.shape, eg_dict['shape'])
            assert_dt_equal(img.get_data_dtype(), eg_dict['dtype'])
            # Check against values from Philips converted nifti image
            data = img.get_data()
            assert_data_similar(data, eg_dict)
            assert_almost_equal(img.header.get_zooms(), eg_dict['zooms'])
            # Standard save does not save extensions
            assert_equal(len(img.header.extensions), 0)
            # Delete previous img, data to make Windows happier
            del img, data
            # Does not overwrite unless option given
            code, stdout, stderr = run_command(
                ['parrec2nii', fname], check_code=False)
            assert_equal(code, 1)
            # Default scaling is dv
            pr_img = load(fname)
            flipped_data = flip_axis(pr_img.get_data(), 1)
            base_cmd = ['parrec2nii', '--overwrite', fname]
            check_conversion(base_cmd, flipped_data, out_froot)
            check_conversion(base_cmd + ['--scaling=dv'],
                             flipped_data,
                             out_froot)
            # fp
            pr_img = load(fname, scaling='fp')
            flipped_data = flip_axis(pr_img.get_data(), 1)
            check_conversion(base_cmd + ['--scaling=fp'],
                             flipped_data,
                             out_froot)
            # no scaling
            unscaled_flipped = flip_axis(pr_img.dataobj.get_unscaled(), 1)
            check_conversion(base_cmd + ['--scaling=off'],
                             unscaled_flipped,
                             out_froot)
            # Save extensions
            run_command(base_cmd + ['--store-header'])
            img = load(out_froot)
            assert_equal(len(img.header.extensions), 1)
            del img  # To help windows delete the file


@script_test
@needs_nibabel_data('nitest-balls1')
def test_parrec2nii_with_data():
    # Use nibabel-data to test conversion
    # Premultiplier to relate our affines to Philips conversion
    LAS2LPS = inv_ornt_aff([[0, 1], [1, -1], [2, 1]], (80, 80, 10))
    with InTemporaryDirectory():
        for par in glob(pjoin(BALLS, 'PARREC', '*.PAR')):
            par_root, ext = splitext(basename(par))
            # NA.PAR appears to be a localizer, with three slices in each of
            # the three orientations: sagittal; coronal, transverse
            if par_root == 'NA':
                continue
            # Do conversion
            run_command(['parrec2nii', par])
            conved_img = load(par_root + '.nii')
            # Confirm parrec2nii conversions are LAS
            assert_equal(aff2axcodes(conved_img.affine), tuple('LAS'))
            # Shape same whether LPS or LAS
            assert_equal(conved_img.shape[:3], (80, 80, 10))
            # Test against original converted NIfTI
            nifti_fname = pjoin(BALLS, 'NIFTI', par_root + '.nii.gz')
            if exists(nifti_fname):
                philips_img = load(nifti_fname)
                # Confirm Philips converted image always LPS
                assert_equal(aff2axcodes(philips_img.affine), tuple('LPS'))
                # Equivalent to Philips LPS affine
                equiv_affine = conved_img.affine.dot(LAS2LPS)
                assert_almost_equal(philips_img.affine[:3, :3],
                                    equiv_affine[:3, :3], 3)
                # The translation part is always off by the same ammout
                aff_off = equiv_affine[:3, 3] - philips_img.affine[:3, 3]
                assert_almost_equal(aff_off, AFF_OFF, 3)
                # The difference is max in the order of 0.5 voxel
                vox_sizes = vox_size(philips_img.affine)
                assert_true(np.all(np.abs(aff_off / vox_sizes) <= 0.5))
                # The data is very close, unless it's the fieldmap
                if par_root != 'fieldmap':
                    conved_data_lps = flip_axis(conved_img.dataobj, 1)
                    assert_true(np.allclose(conved_data_lps,
                                            philips_img.dataobj))
    with InTemporaryDirectory():
        # Test some options
        dti_par = pjoin(BALLS, 'PARREC', 'DTI.PAR')
        run_command(['parrec2nii', dti_par])
        assert_true(exists('DTI.nii'))
        assert_false(exists('DTI.bvals'))
        assert_false(exists('DTI.bvecs'))
        # Does not overwrite unless option given
        code, stdout, stderr = run_command(['parrec2nii', dti_par],
                                           check_code=False)
        assert_equal(code, 1)
        # Writes bvals, bvecs files if asked
        run_command(['parrec2nii', '--overwrite', '--keep-trace',
                     '--bvs', dti_par])
        assert_almost_equal(np.loadtxt('DTI.bvals'), DTI_PAR_BVALS)
        img = load('DTI.nii')
        data = img.get_data().copy()
        del img
        # Bvecs in header, transposed from PSL to LPS
        bvecs_LPS = DTI_PAR_BVECS[:, [2, 0, 1]]
        # Adjust for output flip of Y axis in data and bvecs
        bvecs_LAS = bvecs_LPS * [1, -1, 1]
        assert_almost_equal(np.loadtxt('DTI.bvecs'), bvecs_LAS.T)
        # Dwell time
        assert_false(exists('DTI.dwell_time'))
        # Need field strength if requesting dwell time
        code, _, _, = run_command(
            ['parrec2nii', '--overwrite', '--dwell-time', dti_par],
            check_code=False)
        assert_equal(code, 1)
        run_command(
            ['parrec2nii', '--overwrite', '--dwell-time',
             '--field-strength', '3', dti_par])
        exp_dwell = (26 * 9.087) / (42.576 * 3.4 * 3 * 28)
        with open('DTI.dwell_time', 'rt') as fobj:
            contents = fobj.read().strip()
        assert_almost_equal(float(contents), exp_dwell)
        # ensure trace is removed by default
        run_command(['parrec2nii', '--overwrite', '--bvs', dti_par])
        assert_true(exists('DTI.bvals'))
        assert_true(exists('DTI.bvecs'))
        img = load('DTI.nii')
        bvecs_notrace = np.loadtxt('DTI.bvecs').T
        bvals_notrace = np.loadtxt('DTI.bvals')
        data_notrace = img.get_data().copy()
        assert_equal(data_notrace.shape[-1], len(bvecs_notrace))
        del img
        # ensure correct volume was removed
        good_mask = np.logical_or((bvecs_notrace != 0).any(axis=1),
                                  bvals_notrace == 0)
        assert_almost_equal(data_notrace, data[..., good_mask])
        assert_almost_equal(bvals_notrace, np.array(DTI_PAR_BVALS)[good_mask])
        assert_almost_equal(bvecs_notrace, bvecs_LAS[good_mask])
