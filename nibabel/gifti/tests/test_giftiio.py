# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from ..giftiio import read, write
from .test_parse_gifti_fast import DATA_FILE1

import pytest


def test_read_deprecated(tmp_path):
    with pytest.deprecated_call():
        img = read(DATA_FILE1)

    fname = tmp_path / 'test.gii'
    with pytest.deprecated_call():
        write(img, fname)
