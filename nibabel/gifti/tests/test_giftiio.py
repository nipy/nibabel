# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

try:
    import nibabel.gifti as gi
except ImportError:
    from nose import SkipTest
    raise SkipTest

import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises
     
from os.path import join as pjoin, dirname

IO_DATA_PATH = pjoin(dirname(__file__), 'data')

DATA_FILE1 = pjoin(IO_DATA_PATH, 'ascii.gii')
DATA_FILE2 = pjoin(IO_DATA_PATH, 'rh.shape.curv.gii')

DATA_FILE1_darr = np.array(
       [[-16.07201 , -66.187515,  21.266994],
       [-16.705893, -66.054337,  21.232786],
       [-17.614349, -65.401642,  21.071466]])
     

def test_read():
    a = gi.read(DATA_FILE1)
    gi.write(a, 'data/testout.gii')
    yield assert_array_almost_equal(a.darrays[0].data, DATA_FILE1_darr)
    
def test_read2():
    a = gi.read(DATA_FILE2)
    gi.write(a, 'data/testout.gii')
    yield assert_array_almost_equal(a.darrays[0].data, DATA_FILE1_darr)
    
    
#def test_write():
#    a = gi.read(DATA_FILE1)
#    gi.write(a, 'test.gii')

    

