""" Testing filesets - a draft

"""
import os
from tempfile import mkstemp

from cStringIO import StringIO

import numpy as np

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nibabel.testing import parametric


@parametric
def test_highest():
    # test high-level interface to filesets
    arr = np.zeros((2,3,4))
    aff = np.eye(4)
    img = Nifti1Image(arr, aff)
    # single image
    img.set_filename('test')
    yield assert_equal(img.get_filename(), 'test.nii')
    yield assert_equal(img.files['image'].filename, 'test.nii')
    yield assert_raises(KeyError, img.files.__getitem__, 'header')
    # pair
    img.files = NiftiFilePair()
    img.set_filename('test')
    yield assert_equal(img.get_filename(), 'test.img')
    yield assert_equal(img.files['image'].filename, 'test.img')
    yield assert_equal(img.files['header'].filename, 'test.hdr')
    

def test_interface():
    # test interface for filesets
    types = ('f1', 'f2')
    class C(FileSet):
        types = types
    c = C()
    for key in types:
        yield assert_equal(c.get_filename(key), None)
        yield assert_equal(c.get_fileobj(key), None)
        yield assert_equal(c.get_start_pos(key), 0)
    # fileobj interface
    stio = StringIO()
    stio.write('aaa')
    c.set_fileobj('f1', stio)
    yield assert_equal(c.get_filename('f1'), None)
    yield assert_equal(c.get_fileobj('f1'), stio)
    yield assert_equal(c.get_start_pos('f1'), 3)
    # filename interface
    try:
        fd, fname = mkstemp()
        c.set_filename('f1', fname)
        yield assert_equal(c.get_filename('f1'), fname)
    finally:
        os.remove(fname)


# disable tests for now
test_interface.__test__ = False
test_highest.__test__ = False
