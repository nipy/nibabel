# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from os.path import join as pjoin, dirname

import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises
     

try:
    import nibabel.gifti as gi
except ImportError:
    from nose import SkipTest
    raise SkipTest


IO_DATA_PATH = pjoin(dirname(__file__), 'data')

DATA_FILE1 = pjoin(IO_DATA_PATH, 'ascii.gii')
DATA_FILE2 = pjoin(IO_DATA_PATH, 'gzipbase64.gii')
DATA_FILE3 = pjoin(IO_DATA_PATH, 'label.gii')
DATA_FILE4 = pjoin(IO_DATA_PATH, 'rh.shape.curv.gii')

datafiles = [DATA_FILE1, DATA_FILE2, DATA_FILE3, DATA_FILE4]
numda = [2, 1, 1, 1]
 
DATA_FILE1_darr1 = np.array(
       [[-16.07201 , -66.187515,  21.266994],
       [-16.705893, -66.054337,  21.232786],
       [-17.614349, -65.401642,  21.071466]])
DATA_FILE1_darr2 = None
DATA_FILE2_darr1 = None
DATA_FILE3_darr1 = None
DATA_FILE4_darr1 = None


def test_metadata():
    
    for i, dat in enumerate(datafiles):
        
        img = gi.read(dat)
        me = img.get_metadata()
        medat = me.get_data_as_dict()
        
        print medat
        
        # numDa = numda[i]
#    version = str
    #filename = str
    
    
def test_dataarray():
    pass

    # assert_array_almost_equal(a.darrays[0].data, DATA_FILE1_darr)
    
def test_readwritedata():
    # read data
    # write data
    # read it again and compare
    pass
