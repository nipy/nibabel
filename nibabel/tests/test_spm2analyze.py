''' Tests for SPM2 header stuff '''

from nibabel.spm2analyze import Spm2AnalyzeHeader

from test_spm99analyze import TestSpm99AnalyzeHeader

class TestSpm2AnalyzeHeader(TestSpm99AnalyzeHeader):
    header_class = Spm2AnalyzeHeader


def test_origin_affine():
    # check that origin affine works, only
    hdr = Spm2AnalyzeHeader()
    aff = hdr.get_origin_affine()
