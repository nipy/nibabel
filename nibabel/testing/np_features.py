""" Look for changes in numpy behavior over versions
"""

import numpy as np


def memmap_after_ufunc():
    """ Return True if ufuncs on memmap arrays always return memmap arrays

    This should be True for numpy < 1.12, False otherwise.

    Memoize after first call.  We do this to avoid having to call this when
    importing nibabel.testing, because we cannot depend on the source file
    being present - see gh-571.
    """
    if memmap_after_ufunc.result is not None:
        return memmap_after_ufunc.result
    with open(__file__, 'rb') as fobj:
        mm_arr = np.memmap(fobj, mode='r', shape=(10,), dtype=np.uint8)
        memmap_after_ufunc.result = isinstance(mm_arr + 1, np.memmap)
    return memmap_after_ufunc.result

memmap_after_ufunc.result = None
