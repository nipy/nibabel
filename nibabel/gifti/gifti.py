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
    """ A list of GiftiNVPairs in stored in
    the list self.data """
    
    def __init__(self, nvpair = None):
        self.data = []
        if not nvpair is None:
            self.data.append(nvpair)

    @classmethod
    def from_dict(cls, data_dict):
        meda = GiftiMetaData()
        for k,v in data_dict.items():
            nv = GiftiNVPairs(k, v)
            meda.data.append(nv)
        return meda
        
    def get_data_as_dict(self):
        """ Returns metadata as dictionary """
        
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
    
    def __init__(self, name = '', value = ''):
        self.name = name
        self.value = value

class GiftiLabelTable(object):

    def __init__(self):

        self.labels = []

    def get_labels_as_dict(self):
        self.labels_as_dict = {}
        for ele in self.labels:
            self.labels_as_dict[ele.index] = ele.label
        return self.labels_as_dict

    def to_xml(self):
        res = "<LabelTable>\n"
        for ele in self.labels:
            lab = """\t<Label Index="%s" Red="%s" Green="%s" Blue="%s" Alpha="%s"><![CDATA[%s]]></Label>\n""" % \
                (str(ele.index), str(ele.red), str(ele.green), str(ele.blue), str(ele.alpha), ele.label)
            res = res + lab
        res = res + "</LabelTable>\n" 
        return res
    
    def print_summary(self):
        print self.get_labels_as_dict()


class GiftiLabel(object):

    index = int
    label = str
    # rgba
    # freesurfer examples seem not to conform
    # to datatype "NIFTI_TYPE_RGBA32" because they
    # are floats, not unsigned 32-bit integers
    
    red = float
    green = float
    blue = float
    alpha = float
    
    def __init__(self, index = 0, label = '', red = 0.0,\
                  green = 0.0, blue = 0.0, alpha = 1.0):
        self.index = index
        self.label = label
        
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha
        
    def get_rgba(self):
        """ Returns RGBA as tuple """
        return (self.red, self.green, self.blue, self.alpha)
        
        

class GiftiCoordSystem(object):

    dataspace = int
    xformspace = int
    xform = None # 4x4 numpy array

    def __init__(self, dataspace = 0, xformspace = 0, xform = None):
        
        self.dataspace = dataspace
        self.xformspace = xformspace
        
        if xform is None:
            # create identity matrix
            self.xform = np.identity(4)
        else:
            self.xform = xform

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
    ext_offset = str

    data = None
    coordsys = None # GiftiCoordSystem()
    meta = None # GiftiMetaData()
    
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
        result += self.to_xml_open()
        
        # write metadata
        if not self.meta is None:
            result += self.meta.to_xml()
            
        # write coord sys
        if not self.coordsys is None:
            result += self.coordsys.to_xml()
        
        # write data array depending on the encoding
        result += "<Data></Data>\n"
        
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
                      gifti_encoding_codes.giistring[self.encoding], \
                      gifti_endian_codes.giistring[self.endian], \
                      self.ext_fname,
                      self.ext_offset,
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

    def __init__(self, meta = None, labeltable = None, darrays = [], \
                 version = "1.0"):

        self.darrays = darrays
        
        if meta is None:
            self.meta = GiftiMetaData()
        else:
            self.meta = meta
            
        if labeltable is None:
            self.labeltable = GiftiLabelTable()
        else:
            self.labeltable = labeltable
        
        self.numDA = len(self.darrays)
        self.version = version

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


    def get_metadata(self):
        return self.meta

    def set_metadata(self, meta):
        # XXX type checks
        self.meta = meta
        print "New Metadata set. Be aware of changing coordinate transformation!"

    def add_gifti_data_array(self, dataarr):
        # XXX type checks
        self.darrays.append(dataarr)
        self.numDA += 1


    def remove_gifti_data_array(self, dataarr):
        # XXX type checks
        self.darrays.remove(dataarr)
        self.numDA -= 1
        


    def remove_gifti_data_array_by_intent(self, intent):
        """ Removes all the data arrays with the given intent type """
        intent2remove = intent_codes.code[intent]
        
        for dele in self.darrays:
            if dele.intent == intent2remove:
                self.darrays.remove(dele)
                self.numDA -= 1

    def getArraysFromIntent(self, intent):
        """ Returns a a list of GiftiDataArray elements matching
        the given intent """

        it = intent_codes.code[intent]

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
        
    def to_xml(self):
        
        res = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE GIFTI SYSTEM "http://www.nitrc.org/frs/download.php/115/gifti.dtd">
<GIFTI Version="%s"  NumberOfDataArrays="%s">\n""" % (self.version, str(self.numDA))

        if not self.meta is None:
            res += self.meta.to_xml()
            
        if not self.labeltable is None:
            res += self.labeltable.to_xml()
            
        for dar in self.darrays:
            res += dar.to_xml()
            
        res += "</GIFTI>"
        
        return res
     



