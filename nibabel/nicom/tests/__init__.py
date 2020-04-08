from ...pydicom_compat import have_dicom
import unittest

dicom_test = unittest.skipUnless(have_dicom, "Could not import dicom or pydicom")
