# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
from __future__ import division, print_function, absolute_import

from os.path import join as pjoin, dirname

import numpy as np

from ... import gifti as gi
from ...nifti1 import xform_codes

from ...tmpdirs import InTemporaryDirectory

from numpy.testing import assert_array_equal, assert_array_almost_equal

from nose.tools import (assert_true, assert_false, assert_equal,
                        assert_raises)


IO_DATA_PATH = pjoin(dirname(__file__), 'data')

DATA_FILE1 = pjoin(IO_DATA_PATH, 'ascii.gii')
DATA_FILE2 = pjoin(IO_DATA_PATH, 'gzipbase64.gii')
DATA_FILE3 = pjoin(IO_DATA_PATH, 'label.gii')
DATA_FILE4 = pjoin(IO_DATA_PATH, 'rh.shape.curv.gii')
DATA_FILE5 = pjoin(IO_DATA_PATH, 'base64bin.gii')
DATA_FILE6 = pjoin(IO_DATA_PATH, 'rh.aparc.annot.gii')

datafiles = [DATA_FILE1, DATA_FILE2, DATA_FILE3, DATA_FILE4, DATA_FILE5, DATA_FILE6]
numda = [2, 1, 1, 1, 2, 1]
 
DATA_FILE1_darr1 = np.array(
       [[-16.07201 , -66.187515,  21.266994],
       [-16.705893, -66.054337,  21.232786],
       [-17.614349, -65.401642,  21.071466]])
DATA_FILE1_darr2 = np.array( [0,1,2] )

DATA_FILE2_darr1 = np.array([[ 0.43635699],
       [ 0.270017  ],
       [ 0.133239  ],
       [ 0.35054299],
       [ 0.26538199],
       [ 0.32122701],
       [ 0.23495001],
       [ 0.26671499],
       [ 0.306851  ],
       [ 0.36302799]], dtype=np.float32)

DATA_FILE3_darr1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0])

DATA_FILE4_darr1 = np.array([[-0.57811606],
       [-0.53871965],
       [-0.44602534],
       [-0.56532663],
       [-0.51392376],
       [-0.43225467],
       [-0.54646534],
       [-0.48011276],
       [-0.45624232],
       [-0.31101292]], dtype=np.float32)
       
DATA_FILE6_darr1 = np.array([9182740, 9182740, 9182740], dtype=np.float32)

DATA_FILE5_darr1 = np.array([[ 155.17539978,  135.58103943,   98.30715179],
       [ 140.33973694,  190.0491333 ,   73.24776459],
       [ 157.3598938 ,  196.97969055,   83.65809631],
       [ 171.46174622,  137.43661499,   78.4709549 ],
       [ 148.54592896,   97.06752777,   65.96373749],
       [ 123.45701599,  111.46841431,   66.3571167 ],
       [ 135.30892944,  202.28720093,   36.38148499],
       [ 178.28155518,  162.59469604,   37.75128937],
       [ 178.11087036,  115.28820038,   57.17986679],
       [ 142.81582642,   82.82115173,   31.02205276]], dtype=np.float32)


DATA_FILE5_darr2 = np.array([[ 6402, 17923, 25602],
       [14085, 25602, 17923],
       [25602, 14085,  4483],
       [17923,  1602, 14085],
       [ 4483, 25603, 25602],
       [25604, 25602, 25603],
       [25602, 25604,  6402],
       [25603,  3525, 25604],
       [ 1123, 17922, 12168],
       [25604, 12168, 17922]], dtype=np.int32)

def test_read_ordering():
    # DATA_FILE1 has an expected darray[0].data shape of (3,3).  However if we
    # read another image first (DATA_FILE2) then the shape is wrong
    # Read an image
    img2 = gi.read(DATA_FILE2)
    assert_equal(img2.darrays[0].data.shape, (143479, 1))
    # Read image for which we know output shape
    img = gi.read(DATA_FILE1)
    assert_equal(img.darrays[0].data.shape, (3,3))


def test_metadata():

    for i, dat in enumerate(datafiles):
        
        img = gi.read(dat)
        me = img.get_metadata()
        medat = me.get_metadata()

        assert_equal(numda[i], img.numDA)
        assert_equal(img.version,'1.0')


def test_dataarray1():
    
    img = gi.read(DATA_FILE1)
    assert_array_almost_equal(img.darrays[0].data, DATA_FILE1_darr1)
    assert_array_almost_equal(img.darrays[1].data, DATA_FILE1_darr2)
    
    me=img.darrays[0].meta.get_metadata()
    
    assert_true('AnatomicalStructurePrimary' in me)
    assert_true('AnatomicalStructureSecondary' in me)
    assert_equal(me['AnatomicalStructurePrimary'], 'CortexLeft')
    
    assert_array_almost_equal(img.darrays[0].coordsys.xform, np.eye(4,4))
    assert_equal(xform_codes.niistring[img.darrays[0].coordsys.dataspace],'NIFTI_XFORM_TALAIRACH')
    assert_equal(xform_codes.niistring[img.darrays[0].coordsys.xformspace],'NIFTI_XFORM_TALAIRACH')

def test_dataarray2():
    img2 = gi.read(DATA_FILE2)
    assert_array_almost_equal(img2.darrays[0].data[:10], DATA_FILE2_darr1)
    
def test_dataarray3():
    img3 = gi.read(DATA_FILE3)
    assert_array_almost_equal(img3.darrays[0].data[30:50], DATA_FILE3_darr1)

def test_dataarray4():
    img4 = gi.read(DATA_FILE4)
    assert_array_almost_equal(img4.darrays[0].data[:10], DATA_FILE4_darr1)

def test_dataarray5():
    img3 = gi.read(DATA_FILE5)
    assert_array_almost_equal(img3.darrays[0].data, DATA_FILE5_darr1)
    assert_array_almost_equal(img3.darrays[1].data, DATA_FILE5_darr2)


def test_readwritedata():
    img = gi.read(DATA_FILE2)
    with InTemporaryDirectory():
        gi.write(img, 'test.gii')
        img2 = gi.read('test.gii')
        assert_equal(img.numDA,img2.numDA)
        assert_array_almost_equal(img.darrays[0].data,
                                  img2.darrays[0].data)


def test_newmetadata():
    img = gi.GiftiImage()
    attr = gi.GiftiNVPairs(name = 'mykey', value = 'val1')
    newmeta = gi.GiftiMetaData(attr)
    img.set_metadata(newmeta)
    myme = img.meta.get_metadata()
    assert_true('mykey' in myme)
    newmeta = gi.GiftiMetaData.from_dict( {'mykey1' : 'val2'} )
    img.set_metadata(newmeta)
    myme = img.meta.get_metadata()
    assert_true('mykey1' in myme)
    assert_false('mykey' in myme)


def test_getbyintent():
    img = gi.read(DATA_FILE1)
    da = img.getArraysFromIntent("NIFTI_INTENT_POINTSET")
    assert_equal(len(da), 1)
    da = img.getArraysFromIntent("NIFTI_INTENT_TRIANGLE")
    assert_equal(len(da), 1)
    da = img.getArraysFromIntent("NIFTI_INTENT_CORREL")
    assert_equal(len(da), 0)
    assert_equal(da, [])


def test_labeltable():
    img = gi.read(DATA_FILE6)
    assert_array_almost_equal(img.darrays[0].data[:3], DATA_FILE6_darr1)
    assert_equal(len(img.labeltable.labels), 36)
    labeldict = img.labeltable.get_labels_as_dict()
    assert_true(660700 in labeldict)
    assert_equal(labeldict[660700], 'entorhinal')
    assert_equal(img.labeltable.labels[1].key, 2647065)
    assert_equal(img.labeltable.labels[1].red, 0.0980392)
    assert_equal(img.labeltable.labels[1].green, 0.392157)
    assert_equal(img.labeltable.labels[1].blue, 0.156863)
    assert_equal(img.labeltable.labels[1].alpha, 1)
