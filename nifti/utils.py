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

import nifti
import numpy
import scipy.stats

def time2vol( t, tr, lag=0.0, decimals=0 ):
    """ Translates a time 't' into a volume number. By default function returns
    the volume number that is closest in time. Volumes are assumed to be
    recorded exactly (and completely) after tr/2, e.g. if 'tr' is 2 secs the
    first volume is recorded at exactly one second.

    't' might be a single value, a sequence or an array.

    The repetition 'tr' might be specified directly, but can also be a 
    NiftiImage object. In the latter case the value of 'tr' is determined from
    the 'rtime' property of the NiftiImage object.

    't' and 'tr' can be given in an arbitrary unit (but both have to be in the
    same unit).

    The 'lag' argument can be used to shift the times by constant offset.

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


def applyFxToVolumes( ts, vols, fx, **kwargs ):
    """ Apply a function on selected volumes of a timeseries.

    'ts' is a 4d timeseries. It can be a NiftiImage or a numpy array.
    In case of a numpy array one has to make sure that the time is on the
    first axis. 'ts' can actually be of any dimensionality, but datasets aka
    volumes are assumed to be along the first axis.

    'vols' is either a sequence of sequences or a 2d array indicating which 
    volumes fx should be applied to. Each row defines a set of volumes.

    'fx' is a callable function to get an array of the selected volumes as
    argument. Additonal arguments may be specified as keyword arguments and
    are passed to 'fx'.

    The output will be a 4d array with one computed volume per row in the 'vols'
    array.
    """
    # get data array from nifti image or assume data array is
    # already present
    if isinstance( ts, nifti.NiftiImage ):
        data = ts.data
    else:
        data = ts

    out = []

    for i, vol in enumerate(vols):
        out.append( fx( data[ numpy.array( vol ) ], **kwargs ) )

    return numpy.array( out )


def cropImage( nimg, bbox ):
    """ Crop an image.

    'bbox' has to be a sequency of (min,max) tuples (one for each image
    dimension).

    The function returns the cropped image. The data is not shared with the
    original image, but is copied.
    """

    # build crop command
    cmd = 'nimg.data.squeeze()['
    cmd += ','.join( [ ':'.join( [ str(i) for i in dim ] ) for dim in bbox ] )
    cmd += ']'

    # crop the image data array
    cropped = eval(cmd).copy()

    # return the cropped image with preserved header data
    return nifti.NiftiImage(cropped, nimg.header)


def getPeristimulusTimeseries( ts, onsetvols, nvols = 10):
    """ Returns 4d array with peristimulus timeseries.
    """
    selected = [ [ o + offset for o in onsetvols ] \
                    for offset in range( nvols ) ]

    return applyFxToVolumes(ts, selected, scipy.stats.mean)


def zscore( data, mean = None, std = None ):
    """ Z-Score a dataset.

    'data' can be given as a NiftiImage instance or a NumPy array. By default
    the mean and standard deviation of the data is computed along the first
    axis of the data array.

    'mean' and 'std' can be used to pass custom values to the z-scoring. Both
    may be scalars or arrays.

    All computations are done in-place.
    """
    # get data array from nifti image or assume data array is
    # already present
    if isinstance( data, nifti.NiftiImage ):
        data = data.data

    # calculate mean if necessary
    if not mean:
        mean = data.mean()

    # calculate std-deviation if necessary
    if not std:
        std = data.std()

    # do the z-scoring (do not use in-place operations to ensure
    # appropriate data upcasting
    return ( data - mean ) / std
