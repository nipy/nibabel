""" Adapter module for working with pydicom < 1.0 and >= 1.0

In what follows, "dicom is available" means we can import either a) ``dicom``
(pydicom < 1.0) or or b) ``pydicom`` (pydicom >= 1.0).

Regardless of whether dicom is available this module should be importable
without error, and always defines:

* have_dicom : True if we can import pydicom or dicom;
* pydicom : pydicom module or dicom module or None of not importable;
* read_file : ``read_file`` function if pydicom or dicom module is importable
  else None;
* tag_for_keyword : ``tag_for_keyword`` function if pydicom or dicom module
  is importable else None;

A test decorator is available in nibabel.nicom.tests:

* dicom_test : test decorator that skips test if dicom not available.

A deprecated copy is available here for backward compatibility.
"""

# Module has (apparently) unused imports; stop flake8 complaining
# flake8: noqa

import numpy as np
from .deprecated import deprecate_with_version

have_dicom = True
pydicom = read_file = tag_for_keyword = Sequence = None

try:
    import dicom as pydicom
except ImportError:
    try:
        import pydicom
    except ImportError:
        have_dicom = False
    else:  # pydicom module available
        from pydicom.dicomio import read_file
        from pydicom.sequence import Sequence
        # Values not imported by default
        import pydicom.values
else:  # dicom module available
    # Values not imported by default
    import dicom.values
    from dicom.sequence import Sequence
    read_file = pydicom.read_file

if have_dicom:
    try:
        # Versions >= 1.0
        tag_for_keyword = pydicom.datadict.tag_for_keyword
    except AttributeError:
        # Versions < 1.0 - also has more search options.
        tag_for_keyword = pydicom.datadict.tag_for_name


@deprecate_with_version("dicom_test has been moved to nibabel.nicom.tests",
                        since="3.1", until="5.0")
def dicom_test(func):
    # Import locally to avoid circular dependency
    from .nicom.tests import dicom_test
    return dicom_test(func)
