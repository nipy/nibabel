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
from StringIO import StringIO
import numpy as np


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

    def to_xml(self):
        res = "<MetaData>\n"
        for ele in self.data:
            nvpair = """<MD>
\t<Name><![CDATA[%s]]></Name>
\t<Value><![CDATA[%s]]></Value>
</MD>\n""" % (ele.name, ele.value)
            res = res + nvpair
        res = res + "</MetaData>\n" 
        return res

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

    dataspace = int
    xformspace = int
    xform = None # will be numpy array

    # XXX: implement init? or as get_identity_gifti_coord_system() as method?
    # see trackvis

    def to_xml(self):
        
        res = """
         <CoordinateSystemTransformMatrix>
         <DataSpace><![CDATA[%s]]></DataSpace>
         <TransformedSpace><![CDATA[%s]]></TransformedSpace>
         """ % (xform_codes.niistring[self.dataspace], \
                xform_codes.niistring[self.xformspace])
         
        e = StringIO()
        np.savetxt(e, self.xfrom)
        e.seek(0)
        res = res + "<MatrixData>"
        res = res + e.read()
        e.close()
        res = res + "<MatrixData>"
        res = res + "</CoordinateSystemTransformMatrix>" 
        return res

    def print_summary(self):

        print 'Dataspace: ', xform_codes.niistring[self.dataspace]
        print 'XFormSpace: ', xform_codes.niistring[self.xformspace]
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
    ext_offset = int

    data = None
    coordsys = None # GiftiCoordSystem()
    # XXX: missing metadata
    
    def __init__(self):

        self.dims = []
        self.meta = GiftiMetaData()
        self.ext_fname = ''
        self.ext_offset = ''

    @classmethod
    def from_array(cls, darray, intent, datatype, coordsys):
        
        cda = GiftiDataArray()
        # XXX: to continue
        # create new dataarray
        # setting the properties given or default
        # return objects
#def GiftiDataArray_fromarray(self, data, intent = GiftiIntentCode.NIFTI_INTENT_NONE, encoding=GiftiEncoding.GIFTI_ENCODING_B64GZ, endian = GiftiEndian.GIFTI_ENDIAN_LITTLE):
#    """ Returns as GiftiDataArray from a Numpy array with a given intent code and
#    encoding """
#    pass
        return cda

    def to_xml(self):
        
        result = ""
        result = result + self.to_xml_open()
        
        # write metadata
        
        
        # XXX: write coord sys
        
        
        # write data arrays (loop)
        
        result = result + self.to_xml_close()
        return result

    def to_xml_open(self):
        out = """<DataArray Intent="%s"
\tDataType="%s"
\tArrayIndexingOrder="%s"
\tDimensionality="%s"
%s\tEncoding="%s"
\tEndian="%s"
\tExternalFileName="%s"
\tExternalFileOffset="%s">\n"""
        di = ""
        for i, n in enumerate(self.dims):
            di = di + '\tDim%s=\"%s\"\n' % (str(i), str(n))
             
        return out % (intent_codes.niistring[self.intent], \
                      data_type_codes.niistring[self.datatype], \
                      array_index_order_codes.label[self.ind_ord], \
                      str(self.num_dim), \
                      str(di), \
                      gifti_encoding_codes.code[self.encoding], \
                      gifti_endian_codes.code[self.endian], \
                      self.ext_fname,
                      str(self.ext_offset),
                      )
              
    def to_xml_close(self):
        return "</DataArray>\n"

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

    @classmethod
    def from_array(cls):
        pass
    
    @classmethod
    def from_vertices_and_triangles(cls, vertices, triangles, coordsys = None, \
                                    encoding = GiftiEncoding.GIFTI_ENCODING_B64GZ,\
                                    endian = GiftiEndian.GIFTI_ENDIAN_LITTLE):
        pass
    
#def GiftiImage_fromarray(data, intent = GiftiIntentCode.NIFTI_INTENT_NONE, encoding=GiftiEncoding.GIFTI_ENCODING_B64GZ, endian = GiftiEndian.GIFTI_ENDIAN_LITTLE):
#    """ Returns a GiftiImage from a Numpy array with a given intent code and
#    encoding """
#    pass

#def GiftiImage_fromTriangles(vertices, triangles, cs = None, encoding=GiftiEncoding.GIFTI_ENCODING_B64GZ, endian = GiftiEndian.GIFTI_ENDIAN_LITTLE):
#    """ Returns a GiftiImage from two numpy arrays representing the vertices
#    and the triangles. Additionally defining the coordinate system and encoding """
#    pass


    # XXX: add getter and setter methods?
    # 
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


