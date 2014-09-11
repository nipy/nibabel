# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test scripts

Test running scripts
"""
from __future__ import division, print_function, absolute_import

import os
from os.path import dirname, join as pjoin, abspath
import re

import numpy as np

from ..tmpdirs import InTemporaryDirectory
from ..loadsave import load

from nose.tools import assert_true, assert_not_equal, assert_equal

from numpy.testing import assert_almost_equal

from .scriptrunner import ScriptRunner


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
    return np.sqrt(np.sum(affine[:3,:3] ** 2, axis=0))


@script_test
def test_parrec2nii():
    # Test parrec2nii script
    cmd = ['parrec2nii', '--help']
    code, stdout, stderr = run_command(cmd)
    assert_true(stdout.startswith('Usage'))
    in_fname = pjoin(DATA_PATH, 'phantom_EPI_asc_CLEAR_2_1.PAR')
    out_froot = 'phantom_EPI_asc_CLEAR_2_1.nii'
    with InTemporaryDirectory():
        run_command(['parrec2nii', in_fname])
        img = load(out_froot)
        assert_equal(img.shape, (64, 64, 9, 3))
        assert_equal(img.get_data_dtype(), np.dtype(np.int16))
        # Check against values from Philips converted nifti image
        data = img.get_data()
        assert_true(np.allclose(
            (data.min(), data.max(), data.mean()),
            (0.0, 2299.4110643863678, 194.95876256117265)))
        assert_almost_equal(vox_size(img.get_affine()), (3.75, 3.75, 8))
