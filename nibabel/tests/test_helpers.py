""" Helper functions for tests
"""
from io import BytesIO

import numpy as np

from ..openers import ImageOpener
from ..tmpdirs import InTemporaryDirectory
from ..optpkg import optional_package
_, have_scipy, _ = optional_package('scipy.io')

from nose.tools import assert_true
from numpy.testing import assert_array_equal


def bytesio_filemap(klass):
    """ Return bytes io filemap for this image class `klass` """
    file_map = klass.make_file_map()
    for name, fileholder in file_map.items():
        fileholder.fileobj = BytesIO()
        fileholder.pos = 0
    return file_map


def bytesio_round_trip(img):
    """ Save then load image from bytesio
    """
    klass = img.__class__
    bytes_map = bytesio_filemap(klass)
    img.to_file_map(bytes_map)
    return klass.from_file_map(bytes_map)


def bz2_mio_error():
    """ Return True if writing mat 4 file fails

    Writing an empty string can fail for bz2 objects in python 3.3:

    https://bugs.python.org/issue16828

    This in turn causes scipy to give this error when trying to write bz2 mat
    files.

    This won't cause a problem for scipy releases after Jan 24 2014 because of
    commit 98ef522d99 (in scipy)
    """
    if not have_scipy:
        return True
    import scipy.io

    with InTemporaryDirectory():
        with ImageOpener('test.mat.bz2', 'wb') as fobj:
            try:
                scipy.io.savemat(fobj, {'a': 1}, format='4')
            except ValueError:
                return True
            else:
                return False


def assert_data_similar(arr, params):
    """ Check data is the same if recorded, otherwise check summaries

    Helper function to test image array data `arr` against record in `params`,
    where record can be the array itself, or summary values from the array.

    Parameters
    ----------
    arr : array-like
        Something that results in an array after ``np.asarry(arr)``
    params : mapping
        Mapping that has either key ``data`` with value that is array-like, or
        key ``data_summary`` with value a dict having keys ``min``, ``max``,
        ``mean``
    """
    if 'data' in params:
        assert_array_equal(arr, params['data'])
        return
    summary = params['data_summary']
    real_arr = np.asarray(arr)
    assert_true(np.allclose(
        (real_arr.min(), real_arr.max(), real_arr.mean()),
        (summary['min'], summary['max'], summary['mean'])))
