# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
# Gifti IO
# Stephan Gerhard, May 2010

from util import *

class GiftiMetaData(object):

    def __init__(self):

        # create a list of GiftiNVPairs objects
        # containing the Name and the Value
        self.data = []

    def get_data_as_dict(self):

        self.data_as_dict = {}

        for ele in self.data:
            self.data_as_dict[ele.name] = ele.value

        return self.data_as_dict

    def print_summary(self):
        print self.get_data_as_dict()


class GiftiNVPairs(object):

    name = str
    value = str

class GiftiLabelTable(object):

    def __init__(self):

        self.labels = []

    def get_labels_as_dict(self):

        self.labels_as_dict = {}

        for ele in self.labels:
            self.labels_as_dict[ele.index] = ele.label

        return self.labels_as_dict

    def print_summary(self):
        print self.get_labels_as_dict()


class GiftiLabel(object):

    index = int
    label = str

class GiftiCoordSystem(object):

    dataspace = str
    xformspace = str
    xform = None # will be numpy array

    def print_summary(self):

        print 'Dataspace: ', self.dataspace
        print 'XFormSpace: ', self.xformspace
        print 'Affine Transformation Matrix: \n', self.xform

class GiftiDataArray(object):

    intent = int
    datatype = int
    ind_ord = int
    num_dim = int
    dims = []
    encoding = int
    endian = int
    ext_fname = str
    ext_offset = None

    data = None
    coordsys = None # GiftiCoordSystem()

    def __init__(self):

        self.dims = []
        self.meta = GiftiMetaData()

    def print_summary(self):

        print 'Intent: ', GiftiIntentCode.intents_inv[self.intent]
        print 'DataType: ', GiftiDataType.datatypes_inv[self.datatype]
        print 'ArrayIndexingOrder: ', GiftiArrayIndexOrder.ordering_inv[self.ind_ord]
        print 'Dimensionality: ', self.num_dim
        print 'Dimensions: ', self.dims
        print 'Encoding: ', GiftiEncoding.encodings_inv[self.encoding]
        print 'Endian: ', GiftiEndian.endian_inv[self.endian]
        print 'ExternalFileName: ', self.ext_fname
        print 'ExternalFileOffset: ', self.ext_offset
        if not self.coordsys == None:
            print '----'
            print 'Coordinate System:'
            print self.coordsys.print_summary()

    def get_meta_as_dict(self):
        return self.meta.get_data_as_dict()

class GiftiImage(object):

    numDA = int
    version = str
    filename = str

    def __init__(self):

        # list of GiftiDataArray
        self.darrays = []
        self.meta = GiftiMetaData()
        self.labeltable = GiftiLabelTable()

    # add getter and setter methods?
    def get_metadata(self):

        return self.meta

    def set_metadata(self, meta):

        # XXX: if it exists, make it readonly?
        # similar issues with nifti,
        # especially for GiftiDataArray metadata!
        # e.g. changing transformation matrix

        self.meta = meta

    def add_gifti_data_array(self, dataarr):
        self.darrays.append(dataarr)

        self.numDA += 1

        # XXX sanity checks

    def remove_gifti_data_array(self, dataarr):
        self.darrays.remove(dataarr)
        # XXX update

    def getArraysFromIntent(self, intent):
        """ Returns a a list of GiftiDataArray elements matching
        the given intent """

        # if it is integer do not convert
        if type(intent)=='int':
            it = GiftiIntentCode.intents[intent]
        else:
            it = intent

        return [x for x in self.darrays if x.intent == it]


    def print_summary(self):

        print '----start----'
        print 'Source filename: ', self.filename
        print 'Number of data arrays: ', self.numDA
        print 'Version: ', self.version
        if not self.meta == None:
            print '----'
            print 'Metadata:'
            print self.meta.print_summary()
        if not self.labeltable == None:
            print '----'
            print 'Labeltable:'
            print self.labeltable.print_summary()
        for i, da in enumerate(self.darrays):
            print '----'
            print 'DataArray %s:' % i
            print da.print_summary()
        print '----end----'


##############
# General Gifti Input - Output to the filesystem
##############

def loadImage(filename):
    """ Load a Gifti image from a file """
    import os.path
    if not os.path.exists(filename):
        raise IOError("No such file or directory: '%s'" % filename)
    else:
        import parse_gifti_fast as pg
        giifile = pg.parse_gifti_file(filename)
        return giifile

def saveImage(image, filename):
    """ Save the current image to a new file

    If the image was created using array data (not loaded from a file) one
    has to specify a filename

    Note that the Gifti spec suggests using the following suffixes to your
    filename when saving each specific type of data:

    .gii
        Generic GIFTI File
    .coord.gii
        Coordinates
    .func.gii
        Functional
    .label.gii
        Labels
    .rgba.gii
        RGB or RGBA
    .shape.gii
        Shape
    .surf.gii
        Surface
    .tensor.gii
        Tensors
    .time.gii
        Time Series
    .topo.gii
        Topology
    """

    #if not image.version:
    #   t = pygiftiio.gifticlib_version()
    #   versionstr = t[t.find("version ")+8:t.find(", ")]
    #   float(versionstr) # raise an exception should the format change in the future :-)
    #   image.version = versionstr

        # how to handle gifticlib? because we use pure python independent of the clib

        # do a validation
        # save GiftiImage to filename

    pass


##############
# special purpose GiftiImage / GiftiDataArray creation methods
##############

#def GiftiImage_fromarray(data, intent = GiftiIntentCode.NIFTI_INTENT_NONE, encoding=GiftiEncoding.GIFTI_ENCODING_B64GZ, endian = GiftiEndian.GIFTI_ENDIAN_LITTLE):
#    """ Returns a GiftiImage from a Numpy array with a given intent code and
#    encoding """
#    pass
#
#def GiftiImage_fromTriangles(vertices, triangles, cs = None, encoding=GiftiEncoding.GIFTI_ENCODING_B64GZ, endian = GiftiEndian.GIFTI_ENDIAN_LITTLE):
#    """ Returns a GiftiImage from two numpy arrays representing the vertices
#    and the triangles. Additionally defining the coordinate system and encoding """
#    pass
#
#def GiftiDataArray_fromarray(self, data, intent = GiftiIntentCode.NIFTI_INTENT_NONE, encoding=GiftiEncoding.GIFTI_ENCODING_B64GZ, endian = GiftiEndian.GIFTI_ENDIAN_LITTLE):
#    """ Returns as GiftiDataArray from a Numpy array with a given intent code and
#    encoding """
#    pass
