''' Tests for SPM2 header stuff '''

from nibabel.spm2analyze import Spm2AnalyzeHeader, Spm2AnalyzeImage

import test_spm99analyze

class TestSpm2AnalyzeHeader(test_spm99analyze.TestSpm99AnalyzeHeader):
    header_class = Spm2AnalyzeHeader


class TestSpm2AnalyzeImage(test_spm99analyze.TestSpm99AnalyzeImage):
    # class for testing images
    image_class = Spm2AnalyzeImage
    header_class = Spm2AnalyzeHeader
    

def test_origin_affine():
    # check that origin affine works, only
    hdr = Spm2AnalyzeHeader()
    aff = hdr.get_origin_affine()
