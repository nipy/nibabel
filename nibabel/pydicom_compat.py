""" Adapter module for working with pydicom < 1.0 and >= 1.0

In what follows, "dicom is available" means we can import either a) ``dicom``
(pydicom < 1.0) or or b) ``pydicom`` (pydicom >= 1.0).

Regardless of whether dicom is available this module should be importable
without error, and always defines:

* have_dicom : True if we can import pydicom or dicom;
* pydicom : pydicom module or dicom module or None of not importable;
* read_file : ``read_file`` function if pydicom or dicom module is importable
  else None;
* dicom_test : test decorator that skips test if dicom not available.
"""

# Module does (apparently) unused imports; stop flake8 complaining
# flake8: noqa

import numpy as np

have_dicom = True
read_file = None
pydicom = None

try:
    import dicom as pydicom
    # Values not imported by default
    import dicom.values
except ImportError:
    try:
        import pydicom
    except ImportError:
        have_dicom = False
    else:  # pydicom module available
        from pydicom.dicomio import read_file
        # Values not imported by default
        import pydicom.values
else:  # dicom module available
    read_file = pydicom.read_file


# test decorator that skips test if dicom not available.
dicom_test = np.testing.dec.skipif(not have_dicom,
                                   'could not import dicom or pydicom')
