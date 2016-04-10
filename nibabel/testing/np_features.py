""" Look for changes in numpy behavior over versions
"""

import numpy as np


def _memmap_after_ufunc():
    """ Return True if ufuncs on memmap arrays always return memmap arrays

    This should be True for numpy < 1.12, False otherwise.
    """
    with open(__file__, 'rb') as fobj:
        mm_arr = np.memmap(fobj, mode='r', shape=(10,), dtype=np.uint8)
        mm_preserved = isinstance(mm_arr + 1, np.memmap)
    return mm_preserved


# True if ufunc on memmap always returns a memmap
VIRAL_MEMMAP = _memmap_after_ufunc()
