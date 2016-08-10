""" Tests for the parrec2nii exe code
"""
import imp, numpy, sys, os
from numpy.testing import (assert_almost_equal,
                           assert_array_equal)
from nose.tools import assert_equal
from mock import Mock, MagicMock
from numpy import array as npa
from nibabel.tests.test_parrec import EG_PAR, VARY_PAR
from os.path import dirname, join, isfile, basename
from ..tmpdirs import InTemporaryDirectory
## Possible locations of the parrec2nii executable;
BINDIRS = [join(dirname(dirname(dirname(__file__))), 'bin'), 
            sys.executable,
            join(os.environ['VIRTUALENV'],'bin')]


AN_OLD_AFFINE = numpy.array(
    [[-3.64994708, 0., 1.83564171, 123.66276611],
     [0., -3.75, 0., 115.617],
     [0.86045705, 0., 7.78655376, -27.91161211],
     [0., 0., 0., 1.]])

PAR_AFFINE = numpy.array(
[[  -3.64994708,    0.  ,          1.83564171,  107.63076611],
 [   0.        ,    3.75,          0.        , -118.125     ],
 [   0.86045705,    0.  ,          7.78655376,  -58.25061211],
 [   0.        ,    0.  ,          0.        ,    1.        ]])

def find_parrec2nii():
    for bindir in BINDIRS:
        parrec2niiPath = join(bindir, 'parrec2nii')
        if isfile(parrec2niiPath):
            return parrec2niiPath
    else:
        raise AssertionError('Could not find parrec2nii executable.')

def test_parrec2nii_sets_qform_with_code2():
    """Unit test that ensures that set_qform() is called on the new header.
    """
    parrec2nii = imp.load_source('parrec2nii', find_parrec2nii())
    parrec2nii.verbose.switch = False

    parrec2nii.io_orientation = Mock()
    parrec2nii.io_orientation.return_value = [[0, 1],[1, 1],[2, 1]] # LAS+ 

    parrec2nii.nifti1 = Mock()
    nimg = Mock()
    nhdr = MagicMock()
    nimg.header = nhdr
    parrec2nii.nifti1.Nifti1Image.return_value = nimg

    parrec2nii.pr = Mock()
    pr_img = Mock()
    pr_hdr = Mock()
    pr_hdr.get_data_scaling.return_value = (npa([]), npa([]))
    pr_hdr.get_bvals_bvecs.return_value = (None, None)
    pr_hdr.get_affine.return_value = AN_OLD_AFFINE
    pr_img.header = pr_hdr
    parrec2nii.pr.load.return_value = pr_img

    opts = Mock()
    opts.outdir = None
    opts.scaling = 'off'
    opts.minmax = [1, 1]
    opts.store_header = False
    opts.bvs = False
    opts.vol_info = False
    opts.dwell_time = False

    infile = 'nonexistent.PAR'
    parrec2nii.proc_file(infile, opts)
    nhdr.set_qform.assert_called_with(pr_hdr.get_affine(), code=2)

def test_parrec2nii_save_load_qform_code():
    """Tests that after parrec2nii saves file, it has the qform 'code'
    set to '2', which means 'aligned', so that other software, e.g. FSL
    picks up the qform.
    """
    import nibabel
    parrec2nii = imp.load_source('parrec2nii', find_parrec2nii())
    parrec2nii.verbose.switch = False

    opts = Mock()
    opts.outdir = None
    opts.scaling = 'off'
    opts.minmax = [1, 1]
    opts.store_header = False
    opts.bvs = False
    opts.vol_info = False
    opts.dwell_time = False
    opts.compressed = False

    with InTemporaryDirectory() as pth:
        opts.outdir = pth
        for fname in [EG_PAR, VARY_PAR]:
            parrec2nii.proc_file(fname, opts)
            outfname = join(pth, basename(fname)).replace('.PAR', '.nii')
            assert isfile(outfname)
            img = nibabel.load(outfname)
            assert_almost_equal(img.get_affine(), PAR_AFFINE, 4)
            assert_array_equal(img.header['qform_code'], 2)
            assert_array_equal(img.header['sform_code'], 2)

