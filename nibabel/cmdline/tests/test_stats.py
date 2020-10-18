#!python
# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from io import StringIO
import sys
import numpy as np

from nibabel.loadsave import save
from nibabel.cmdline.stats import main
from nibabel import Nifti1Image


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


def test_volume(tmpdir):
    mask_data = np.zeros((20, 20, 20), dtype='u1')
    mask_data[5:15, 5:15, 5:15] = 1
    img = Nifti1Image(mask_data, np.eye(4))

    infile = tmpdir / "input.nii"
    save(img, infile)

    args = (f"{infile} --Volume")
    with Capturing() as vol_mm3:
        main(args.split())
    args = (f"{infile} --Volume --units vox")
    with Capturing() as vol_vox:
        main(args.split())

    assert float(vol_mm3[0]) == 1000.0
    assert int(vol_vox[0]) == 1000