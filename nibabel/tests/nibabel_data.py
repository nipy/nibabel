""" Functions / decorators for finding / requiring nibabel-data directory
"""

from os import environ
from os.path import dirname, realpath, join as pjoin, isdir, exists

from numpy.testing.decorators import skipif


def get_nibabel_data():
    """ Return path to nibabel-data or None if missing

    First use ``NIBABEL_DATA_DIR`` environment variable.

    If this variable is missing then look for data in directory below package
    directory.
    """
    nibabel_data = environ.get('NIBABEL_DATA_DIR')
    if nibabel_data is None:
        mod = __import__('nibabel')
        containing_path = dirname(dirname(realpath(mod.__file__)))
        nibabel_data = pjoin(containing_path, 'nibabel-data')
    return nibabel_data if isdir(nibabel_data) else None


def needs_nibabel_data(subdir = None):
    """ Decorator for tests needing nibabel-data

    Parameters
    ----------
    subdir : None or str
        Subdirectory we need in nibabel-data directory.  If None, only require
        nibabel-data directory itself.

    Returns
    -------
    skip_dec : decorator
        Decorator skipping tests if required directory not present
    """
    nibabel_data = get_nibabel_data()
    if nibabel_data is None:
        return skipif(True, "Need nibabel-data directory for this test")
    if subdir is None:
        return skipif(False)
    required_path = pjoin(nibabel_data, subdir)
    return skipif(not exists(required_path),
                  "Need {0} for these tests".format(required_path))
