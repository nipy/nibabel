''' Tests for SPM2 header stuff '''

from nifti.spm2analyze import Spm2AnalyzeHeader

from nifti.tests.test_spm99analyze import TestSpm99AnalyzeHeader

class TestSpm2AnalyzeHeader(TestSpm99AnalyzeHeader):
    header_class = Spm2AnalyzeHeader
