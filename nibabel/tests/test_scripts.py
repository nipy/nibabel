# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Test scripts

If we appear to be running from the development directory, use the scripts in
the top-level folder ``scripts``.  Otherwise try and get the scripts from the
path
"""
from __future__ import division, print_function, absolute_import

import sys
import os
from os.path import dirname, join as pjoin, isfile, isdir, abspath, realpath, pardir
import re

from subprocess import Popen, PIPE

import numpy as np

from ..tmpdirs import InTemporaryDirectory
from ..loadsave import load

from nose.tools import assert_true, assert_not_equal, assert_equal

from numpy.testing import assert_almost_equal

def script_test(func):
    # Decorator to label test as a script_test
    func.script_test = True
    return func
script_test.__test__ = False # It's not a test

# Need shell to get path to correct executables
USE_SHELL = True

DEBUG_PRINT = os.environ.get('NIPY_DEBUG_PRINT', False)

DATA_PATH = abspath(pjoin(dirname(__file__), 'data'))
IMPORT_PATH = abspath(pjoin(dirname(__file__), pardir, pardir))

def local_script_dir(script_sdir):
    # Check for presence of scripts in development directory.  ``realpath``
    # checks for the situation where the development directory has been linked
    # into the path.
    below_us_2 = realpath(pjoin(dirname(__file__), '..', '..'))
    devel_script_dir = pjoin(below_us_2, script_sdir)
    if isfile(pjoin(below_us_2, 'setup.py')) and isdir(devel_script_dir):
        return devel_script_dir
    return None

LOCAL_SCRIPT_DIR = local_script_dir('bin')

def run_command(cmd):
    if LOCAL_SCRIPT_DIR is None:
        env = None
    else: # We are running scripts local to the source tree (not installed)
        # Windows can't run script files without extensions natively so we need
        # to run local scripts (no extensions) via the Python interpreter.  On
        # Unix, we might have the wrong incantation for the Python interpreter
        # in the hash bang first line in the source file.  So, either way, run
        # the script through the Python interpreter
        cmd = "%s %s" % (sys.executable, pjoin(LOCAL_SCRIPT_DIR, cmd))
        # If we're testing local script files, point subprocess to consider
        # current nibabel in favor of possibly installed different version
        env = {'PYTHONPATH': '%s:%s'
               % (IMPORT_PATH, os.environ.get('PYTHONPATH', ''))}
    if DEBUG_PRINT:
        print("Running command '%s'" % cmd)
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=USE_SHELL,
                 env=env)
    stdout, stderr = proc.communicate()
    if proc.poll() == None:
        proc.terminate()
    if proc.returncode != 0:
        raise RuntimeError('Command "%s" failed with stdout\n%s\nstderr\n%s\n'
                           % (cmd, stdout, stderr))
    return proc.returncode, stdout, stderr


def _proc_stdout(stdout):
    stdout_str = stdout.decode('latin1').strip()
    return stdout_str.replace(os.linesep, '\n')


@script_test
def test_nib_ls():
    # test nib-ls script
    fname = pjoin(DATA_PATH, 'example4d.nii.gz')
    expected_re = (" (int16|[<>]i2) \[128,  96,  24,   2\] "
                   "2.00x2.00x2.20x2000.00  #exts: 2 sform$")
    # Need to quote out path in case it has spaces
    cmd = 'nib-ls "%s"'  % (fname)
    code, stdout, stderr = run_command(cmd)
    res = _proc_stdout(stdout)
    assert_equal(fname, res[:len(fname)])
    assert_not_equal(re.match(expected_re, res[len(fname):]), None)


@script_test
def test_nib_nifti_dx():
    # Test nib-nifti-dx script
    clean_hdr = pjoin(DATA_PATH, 'nifti1.hdr')
    cmd = 'nib-nifti-dx "%s"'  % (clean_hdr,)
    code, stdout, stderr = run_command(cmd)
    assert_equal(stdout.strip().decode('latin1'), 'Header for "%s" is clean' % clean_hdr)
    dirty_hdr = pjoin(DATA_PATH, 'analyze.hdr')
    cmd = 'nib-nifti-dx "%s"'  % (dirty_hdr,)
    code, stdout, stderr = run_command(cmd)
    expected = """Picky header check output for "%s"

pixdim[0] (qfac) should be 1 (default) or -1
magic string "" is not valid
sform_code 11776 not valid""" % (dirty_hdr,)
    # Split strings to remove line endings
    assert_equal(_proc_stdout(stdout), expected)


def vox_size(affine):
    return np.sqrt(np.sum(affine[:3,:3] ** 2, axis=0))


@script_test
def test_parrec2nii():
    # Test parrec2nii script
    cmd = 'parrec2nii --help'
    code, stdout, stderr = run_command(cmd)
    stdout = stdout.decode('latin1')
    assert_true(stdout.startswith('Usage'))
    in_fname = pjoin(DATA_PATH, 'phantom_EPI_asc_CLEAR_2_1.PAR')
    out_froot = 'phantom_EPI_asc_CLEAR_2_1.nii'
    with InTemporaryDirectory():
        run_command('parrec2nii "{0}"'.format(in_fname))
        img = load(out_froot)
        assert_equal(img.shape, (64, 64, 9, 3))
        assert_equal(img.get_data_dtype(), np.dtype(np.int16))
        # Check against values from Philips converted nifti image
        data = img.get_data()
        assert_true(np.allclose(
            (data.min(), data.max(), data.mean()),
            (0.0, 2299.4110643863678, 194.95876256117265)))
        assert_almost_equal(vox_size(img.get_affine()), (3.75, 3.75, 8))
