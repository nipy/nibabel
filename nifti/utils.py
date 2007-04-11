### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#
#    Utility function for PyNifti
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    version 2 of the License, or (at your option) any later version.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# SVN version control block - do not edit manually
# $Id$
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

import nifti
import numpy

def time2vol( t, tr, lag=0.0, decimals=0 ):
    """ Translates a time 't' into a volume number. By default function returns
    the volume number that is closest in time. Volumes are assumed to be
    recorded exactly (and completely) after tr/2, e.g. if 'tr' is 2 secs the
    first volume is recorded after one second.

    't' might be a single value a sequence or an array.

    The repetition 'tr' might be specified directly, but can also be a 
    NiftiImage object. In the latter case the value of 'tr' is determined from
    the 'rtime' property of the NiftiImage object.

    't' and 'tr' can be given in an arbitrary unit (but both in the same unit).

    The 'lag' argument can be used to shift the times be constant offset.

    Please note that numpy.round() is used to round to interger value (rounds
    to even numbers). The 'decimals' argument will be passed to numpy.round().
    """
    # transform to numpy array for easy handling
    tmp = numpy.array(t)
    
    # determine tr if NiftiImage object
    if isinstance( tr, nifti.NiftiImage ):
        tr = tr.rtime

    vol = numpy.round( ( tmp + lag + tr/2 ) / tr, decimals )

    return vol


def calcConditionMeans( ts, vols ):
    """ Compute the mean of a series of volumes.

    'ts' is a 4d timeseries. It can be a NiftiImage or a numpy array.
    In case of a numpy array one has to make sure that the time is on the first
    axis.

    'vols' is either a sequence of sequences or a 2d array indicating which 
    volumes are to be averaged. Each row defines volumes that are to be averaged.

    The output will be a 4d array with one average volume per row in the 'vols'
    array.
    """
    # get data array from nifti image or assume data array is
    # already present
    if isinstance( ts, nifti.NiftiImage ):
        data = ts.data
    else:
        data = ts

    # check for 4d array
    if not len( data.shape ) == 4:
        raise ValueError, 'This function only handles 4d arrays'

    # determine the size of the output image
    # input data is assumed to have time on the first axis
    avg = numpy.zeros( (len(vols),) + data.shape[1:4] )

    for i, vol in enumerate(vols):
        # calc mean of all indicated volumes
        for v in vol:
            avg[i] += data[v]

        avg[i] /= len(vol)

    return avg


def cropImage( nimg, bbox ):
    """ Crop an image.

    'bbox' has to be a sequency of (min,max) tuples (one for each image
    dimension).

    The function returns the cropped image. The data is not shared with the
    original image, but is a copy.
    """

    # build crop command
    cmd = 'nimg.data.squeeze()['
    cmd += ','.join( [ ':'.join( [ str(i) for i in dim ] ) for dim in bbox ] )
    cmd += ']'

    # crop the image data array
    cropped = eval(cmd).copy()

    # return the cropped image with preserved header data
    return nifti.NiftiImage(cropped, nimg.header)

