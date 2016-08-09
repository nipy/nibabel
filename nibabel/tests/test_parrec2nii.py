""" Tests for the parrec2nii exe code
"""
import imp, numpy
from mock import Mock, MagicMock
from numpy import array as npa

AN_OLD_AFFINE = numpy.array(
    [[-3.64994708, 0., 1.83564171, 123.66276611],
     [0., -3.75, 0., 115.617],
     [0.86045705, 0., 7.78655376, -27.91161211],
     [0., 0., 0., 1.]])

def test_parrec2nii():
    parrec2nii = imp.load_source('parrec2nii', 'bin/parrec2nii')
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
