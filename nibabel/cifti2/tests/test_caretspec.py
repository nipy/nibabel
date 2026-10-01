import unittest
from pathlib import Path

from nibabel.cifti2.caretspec import *
from nibabel.optpkg import optional_package
from nibabel.testing import data_path

requests, has_requests, _ = optional_package('requests')


def test_CaretSpecFile():
    fsLR = CaretSpecFile.from_filename(Path(data_path) / 'fsLR.wb.spec')

    assert fsLR.metadata == {}
    assert fsLR.version == '1.0'
    assert len(fsLR.data_files) == 5

    for df in fsLR.data_files:
        assert isinstance(df, CaretSpecDataFile)
        if df.data_file_type == 'SURFACE':
            assert isinstance(df, SurfaceDataFile)


@unittest.skipUnless(has_requests, reason='Test fetches from URL')
def test_SurfaceDataFile():
    fsLR = CaretSpecFile.from_filename(Path(data_path) / 'fsLR.wb.spec')
    df = fsLR.data_files[0]
    assert df.data_file_type == 'SURFACE'
    try:
        coords, triangles = df.get_mesh()
    except IOError:
        raise unittest.SkipTest(reason='Broken URL')
    assert coords.shape == (32492, 3)
    assert triangles.shape == (64980, 3)
