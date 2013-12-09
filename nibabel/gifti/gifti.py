# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
from __future__ import division, print_function, absolute_import

import sys

from ..externals.six import BytesIO

import numpy as np

from ..nifti1 import data_type_codes, xform_codes, intent_codes
from .util import (array_index_order_codes, gifti_encoding_codes,
                   gifti_endian_codes, KIND2FMT)

# {en,de}codestring in deprecated in Python3, but
# {en,de}codebytes not available in Python2. 
# Therefore set the proper functions depending on the Python version.
import base64
if sys.version < '3':
    base64_encodebytes = base64.encodestring
    base64_decodebytes = base64.decodestring
else:
    base64_encodebytes = base64.encodebytes
    base64_decodebytes = base64.decodebytes
                   

class GiftiMetaData(object):
    """ A list of GiftiNVPairs in stored in
    the list self.data """
    def __init__(self, nvpair = None):
        self.data = []
        if not nvpair is None:
            self.data.append(nvpair)

    @classmethod
    def from_dict(klass, data_dict):
        meda = klass()
        for k,v in data_dict.items():
            nv = GiftiNVPairs(k, v)
            meda.data.append(nv)
        return meda

    def get_metadata(self):
        """ Returns metadata as dictionary """
        self.data_as_dict = {}
        for ele in self.data:
            self.data_as_dict[ele.name] = ele.value
        return self.data_as_dict

    def to_xml(self):
        if len(self.data) == 0:
            return "<MetaData/>\n"
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
        print(self.get_metadata())


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
            self.labels_as_dict[ele.key] = ele.label
        return self.labels_as_dict

    def to_xml(self):
        if len(self.labels) == 0:
            return "<LabelTable/>\n"
        res = "<LabelTable>\n"
        for ele in self.labels:
            col = ''
            if not ele.red is None:
                col += ' Red="%s"' % str(ele.red)
            if not ele.green is None:
                col += ' Green="%s"' % str(ele.green)
            if not ele.blue is None:
                col += ' Blue="%s"' % str(ele.blue)
            if not ele.alpha is None:
                col += ' Alpha="%s"' % str(ele.alpha)
            lab = """\t<Label Key="%s"%s><![CDATA[%s]]></Label>\n""" % \
                (str(ele.key), col, ele.label)
            res = res + lab
        res = res + "</LabelTable>\n" 
        return res

    def print_summary(self):
        print(self.get_labels_as_dict())


class GiftiLabel(object):
    key = int
    label = str
    # rgba
    # freesurfer examples seem not to conform
    # to datatype "NIFTI_TYPE_RGBA32" because they
    # are floats, not unsigned 32-bit integers
    red = float
    green = float
    blue = float
    alpha = float

    def __init__(self, key = 0, label = '', red = None,\
                  green = None, blue = None, alpha = None):
        self.key = key
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
    xform = np.ndarray # 4x4 numpy array

    def __init__(self, dataspace = 0, xformspace = 0, xform = None):
        self.dataspace = dataspace
        self.xformspace = xformspace
        if xform is None:
            # create identity matrix
            self.xform = np.identity(4)
        else:
            self.xform = xform

    def to_xml(self):
        if self.xform is None:
            return "<CoordinateSystemTransformMatrix/>\n"
        res = ("""<CoordinateSystemTransformMatrix>
\t<DataSpace><![CDATA[%s]]></DataSpace>
\t<TransformedSpace><![CDATA[%s]]></TransformedSpace>\n"""
               % (xform_codes.niistring[self.dataspace],
                  xform_codes.niistring[self.xformspace]))
        e = BytesIO()
        np.savetxt(e, self.xform, '%10.6f')
        e.seek(0)
        res = res + "<MatrixData>\n"
        res = res + e.read().decode()
        e.close()
        res = res + "</MatrixData>\n"
        res = res + "</CoordinateSystemTransformMatrix>\n" 
        return res

    def print_summary(self):
        print('Dataspace: ', xform_codes.niistring[self.dataspace])
        print('XFormSpace: ', xform_codes.niistring[self.xformspace])
        print('Affine Transformation Matrix: \n', self.xform)


def data_tag(dataarray, encoding, datatype, ordering):
    """ Creates the data tag depending on the required encoding """
    import zlib
    ord = array_index_order_codes.npcode[ordering]
    enclabel = gifti_encoding_codes.label[encoding]
    if enclabel == 'ASCII':
        c = BytesIO()
        # np.savetxt(c, dataarray, format, delimiter for columns)
        np.savetxt(c, dataarray, datatype, ' ')
        c.seek(0)
        da = c.read()
    elif enclabel == 'B64BIN':
        da = base64_encodebytes(dataarray.tostring(ord))
    elif enclabel == 'B64GZ':
        # first compress
        comp = zlib.compress(dataarray.tostring(ord))
        da = base64_encodebytes(comp)
        da = da.decode()
    elif enclabel == 'External':
        raise NotImplementedError("In what format are the external files?")
    else:
        da = ''
    return "<Data>"+da+"</Data>\n"


class GiftiDataArray(object):

    # These are for documentation only; we don't use these class variables
    intent = int
    datatype = int
    ind_ord = int
    num_dim = int
    dims = list
    encoding = int
    endian = int
    ext_fname = str
    ext_offset = str
    data = np.ndarray
    coordsys = GiftiCoordSystem
    meta = GiftiMetaData

    def __init__(self, data=None):
        self.data = data
        self.dims = []
        self.meta = GiftiMetaData()
        self.coordsys = GiftiCoordSystem()
        self.ext_fname = ''
        self.ext_offset = ''

    @classmethod
    def from_array(klass,
                   darray,
                   intent,
                   datatype = None,
                   encoding = "GIFTI_ENCODING_B64GZ",
                   endian = sys.byteorder,
                   coordsys = None,
                   ordering = "C",
                   meta = None):
        """ Creates a new Gifti data array

        Parameters
        ----------
        darray : ndarray
            NumPy data array
        intent : string
            NIFTI intent code, see nifti1.intent_codes
        datatype : None or string, optional
            NIFTI data type codes, see nifti1.data_type_codes
            If None, the datatype of the NumPy array is taken.
        encoding : string, optionaal
            Encoding of the data, see util.gifti_encoding_codes;
            default: GIFTI_ENCODING_B64GZ
        endian : string, optional
            The Endianness to store the data array.  Should correspond to the
            machine endianness.  default: system byteorder
        coordsys : GiftiCoordSystem, optional
            If None, a identity transformation is taken.
        ordering : string, optional
            The ordering of the array. see util.array_index_order_codes;
            default: RowMajorOrder - C ordering
        meta : None or dict, optional
            A dictionary for metadata information.  If None, gives empty dict.

        Returns
        -------
        da : instance of our own class
        """
        if meta is None:
            meta = {}
        cda = klass(darray)
        cda.num_dim = len(darray.shape)
        cda.dims = list(darray.shape)
        if datatype == None:
            cda.datatype = data_type_codes.code[darray.dtype]
        else:
            cda.datatype = data_type_codes.code[datatype]
        cda.intent = intent_codes.code[intent]
        cda.encoding = gifti_encoding_codes.code[encoding]
        cda.endian = gifti_endian_codes.code[endian]
        if not coordsys is None:
            cda.coordsys = coordsys
        cda.ind_ord = array_index_order_codes.code[ordering]
        cda.meta = GiftiMetaData.from_dict(meta)
        return cda

    def to_xml(self):
        # fix endianness to machine endianness
        self.endian = gifti_endian_codes.code[sys.byteorder]
        result = ""
        result += self.to_xml_open()
        # write metadata
        if not self.meta is None:
            result += self.meta.to_xml()
        # write coord sys
        if not self.coordsys is None:
            result += self.coordsys.to_xml()
        # write data array depending on the encoding
        dt_kind = data_type_codes.dtype[self.datatype].kind
        result += data_tag(self.data,
                           gifti_encoding_codes.specs[self.encoding],
                           KIND2FMT[dt_kind],
                           self.ind_ord)
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
                      gifti_encoding_codes.specs[self.encoding], \
                      gifti_endian_codes.giistring[self.endian], \
                      self.ext_fname,
                      self.ext_offset,
                      )

    def to_xml_close(self):
        return "</DataArray>\n"

    def print_summary(self):
        print('Intent: ', intent_codes.niistring[self.intent])
        print('DataType: ', data_type_codes.niistring[self.datatype])
        print('ArrayIndexingOrder: ',
              array_index_order_codes.label[self.ind_ord])
        print('Dimensionality: ', self.num_dim)
        print('Dimensions: ', self.dims)
        print('Encoding: ', gifti_encoding_codes.specs[self.encoding])
        print('Endian: ', gifti_endian_codes.giistring[self.endian])
        print('ExternalFileName: ', self.ext_fname)
        print('ExternalFileOffset: ', self.ext_offset)
        if not self.coordsys == None:
            print('----')
            print('Coordinate System:')
            print(self.coordsys.print_summary())

    def get_metadata(self):
        """ Returns metadata as dictionary """
        return self.meta.get_metadata()


class GiftiImage(object):

    numDA = int
    version = str
    filename = str

    def __init__(self, meta = None, labeltable = None, darrays = None,
                 version = "1.0"):
        if darrays is None:
            darrays = []
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

#    @classmethod
#    def from_array(cls):
#        pass
#def GiftiImage_fromarray(data, intent = GiftiIntentCode.NIFTI_INTENT_NONE, encoding=GiftiEncoding.GIFTI_ENCODING_B64GZ, endian = GiftiEndian.GIFTI_ENDIAN_LITTLE):
#    """ Returns a GiftiImage from a Numpy array with a given intent code and
#    encoding """


#    @classmethod
#    def from_vertices_and_triangles(cls):
#        pass
#    def from_vertices_and_triangles(cls, vertices, triangles, coordsys = None, \
#                                    encoding = GiftiEncoding.GIFTI_ENCODING_B64GZ,\
#                                    endian = GiftiEndian.GIFTI_ENDIAN_LITTLE):
#    """ Returns a GiftiImage from two numpy arrays representing the vertices
#    and the triangles. Additionally defining the coordinate system and encoding """


    def get_labeltable(self):
        return self.labeltable

    def set_labeltable(self, labeltable):
        """ Set the labeltable for this GiftiImage

        Parameters
        ----------
        labeltable : GiftiLabelTable

        """
        if isinstance(labeltable, GiftiLabelTable):
            self.labeltable = labeltable
        else:
            print("Not a valid GiftiLabelTable instance")

    def get_metadata(self):
        return self.meta

    def set_metadata(self, meta):
        """ Set the metadata for this GiftiImage

        Parameters
        ----------
        meta : GiftiMetaData

        Returns
        -------
        None
        """
        if isinstance(meta, GiftiMetaData):
            self.meta = meta
            print("New Metadata set. Be aware of changing "
                  "coordinate transformation!")
        else:
            print("Not a valid GiftiMetaData instance")

    def add_gifti_data_array(self, dataarr):
        """ Adds a data array to the GiftiImage

        Parameters
        ----------
        dataarr : GiftiDataArray
        """
        if isinstance(dataarr, GiftiDataArray):
            self.darrays.append(dataarr)
            self.numDA += 1
        else:
            print("dataarr paramater must be of tzpe GiftiDataArray")

    def remove_gifti_data_array(self, ith):
        """ Removes the ith data array element from the GiftiImage """
        self.darrays.pop(ith)
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
        print('----start----')
        print('Source filename: ', self.filename)
        print('Number of data arrays: ', self.numDA)
        print('Version: ', self.version)
        if not self.meta == None:
            print('----')
            print('Metadata:')
            print(self.meta.print_summary())
        if not self.labeltable == None:
            print('----')
            print('Labeltable:')
            print(self.labeltable.print_summary())
        for i, da in enumerate(self.darrays):
            print('----')
            print('DataArray %s:' % i)
            print(da.print_summary())
        print('----end----')

    def to_xml(self):
        """ Return XML corresponding to image content """
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

