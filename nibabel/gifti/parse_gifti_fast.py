# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import base64
from sys import byteorder
import zlib
from StringIO import StringIO
from xml.parsers.expat import ParserCreate, ExpatError

import numpy as np

from . import gifti as gi
from .util import (data_type_codes, xform_codes, intent_codes,
                   array_index_order_codes, gifti_encoding_codes,
                   gifti_endian_codes)


DEBUG_PRINT = False


def read_data_block(encoding, endian, ordering, datatype, shape, data):
    """ Tries to unzip, decode, parse the funny string data """
    if ordering == 1:
        ord = 'C'
    elif ordering == 2:
        ord = 'F'
    else:
        ord = 'C'

    if encoding == 1:
        # GIFTI_ENCODING_ASCII
        c = StringIO(data)
        da = np.loadtxt(c)
        da = da.astype(data_type_codes.type[datatype])
        # independent of the endianness
        return da

    elif encoding == 2:
        # GIFTI_ENCODING_B64BIN
        dec = base64.decodestring(data.encode('ascii'))
        dt = data_type_codes.type[datatype]
        sh = tuple(shape)
        newarr = np.fromstring(dec, dtype = dt)
        if len(newarr.shape) != len(sh):
            newarr = newarr.reshape(sh, order = ord)

    elif encoding == 3:
        # GIFTI_ENCODING_B64GZ
        # convert to bytes array for python 3.2
        # http://diveintopython3.org/strings.html#byte-arrays
        dec = base64.decodestring(data.encode('ascii'))
        zdec = zlib.decompress(dec)
        dt = data_type_codes.type[datatype]
        sh = tuple(shape)
        newarr = np.fromstring(zdec, dtype = dt)
        if len(newarr.shape) != len(sh):
            newarr = newarr.reshape(sh, order = ord)

    elif encoding == 4:
        # GIFTI_ENCODING_EXTBIN
        raise NotImplementedError("In what format are the external files?")
    else:
        return 0
        
    # check if we need to byteswap
    # if given endian encoding matches the encoding on the current machine
    # do nothing, otherwise byteswap
    # (1 is big/mac, 2 is little/other)
    print("Endian", endian, "machine", byteorder)
    print("Before")
    print(newarr)
    if ( (endian == 1) and byteorder != 'big' ) or ( (endian == 2) and byteorder != 'little' ):
        print("We byte swap")
        newarr = newarr.byteswap()
    else:
        print("No byte swap")

    print("After")
    print(newarr)
            
    return newarr
    

class Outputter(object):

    def __init__(self):
        self.initialize()

    def initialize(self):
        """ Initialize outputter
        """
        # finite state machine stack
        self.fsm_state = []

        # temporary constructs
        self.nvpair = None
        self.da = None
        self.coordsys = None
        self.lata = None
        self.label = None

        self.meta_global = None
        self.meta_da = None
        self.count_da = True

        # where to write CDATA:
        self.write_to = None
        self.img = None
    
    def StartElementHandler(self, name, attrs):
        if DEBUG_PRINT:
            print 'Start element:\n\t', repr(name), attrs
        
        if name == 'GIFTI':
            # create gifti image
            self.img = gi.GiftiImage()
            if 'Version' in attrs:
                self.img.version = attrs['Version']
            if 'NumberOfDataArrays' in attrs:
                self.img.numDA = int(attrs['NumberOfDataArrays'])
                self.count_da = False

            self.fsm_state.append('GIFTI')

        elif name == 'MetaData':
            self.fsm_state.append('MetaData')

            # if this metadata tag is first, create self.img.meta
            if len(self.fsm_state) == 2:
                self.meta_global = gi.GiftiMetaData()
            else:
                # otherwise, create darray.meta
                self.meta_da = gi.GiftiMetaData()
                

        elif name == 'MD':
            self.nvpair = gi.GiftiNVPairs()
            self.fsm_state.append('MD')

        elif name == 'Name':
            if self.nvpair == None:
                raise ExpatError
            else:
                self.write_to = 'Name'

        elif name == 'Value':
            if self.nvpair == None:
                raise ExpatError
            else:
                self.write_to = 'Value'

        elif name == 'LabelTable':
            self.lata = gi.GiftiLabelTable()
            self.fsm_state.append('LabelTable')

        elif name == 'Label':
            self.label = gi.GiftiLabel()
            if "Index" in attrs:
                self.label.index = int(attrs["Index"])
            if "Red" in attrs:
                self.label.red = float(attrs["Red"])
            if "Green" in attrs:
                self.label.green = float(attrs["Green"])
            if "Blue" in attrs:
                self.label.blue = float(attrs["Blue"])
            if "Alpha" in attrs:
                self.label.alpha = float(attrs["Alpha"])
            
            self.write_to = 'Label'

        elif name == 'DataArray':
            self.da = gi.GiftiDataArray()
            if "Intent" in attrs:
                self.da.intent = intent_codes.code[attrs["Intent"]]
            if "DataType" in attrs:
                self.da.datatype = data_type_codes.code[attrs["DataType"]]
            if "ArrayIndexingOrder" in attrs:
                self.da.ind_ord = array_index_order_codes.code[attrs["ArrayIndexingOrder"]]
            if "Dimensionality" in attrs:
                self.da.num_dim = int(attrs["Dimensionality"])
            for i in range(self.da.num_dim):
                di = "Dim%s" % str(i)
                if di in attrs:
                    self.da.dims.append(int(attrs[di]))

            # dimensionality has to correspond to the number of DimX given
            assert len(self.da.dims) == self.da.num_dim
            if "Encoding" in attrs:
                self.da.encoding = gifti_encoding_codes.code[attrs["Encoding"]]
            if "Endian" in attrs:
                self.da.endian = gifti_endian_codes.code[attrs["Endian"]]
            if "ExternalFileName" in attrs:
                self.da.ext_fname = attrs["ExternalFileName"]
            if "ExternalFileOffset" in attrs:
                self.da.ext_offset = attrs["ExternalFileOffset"]
            
            self.img.darrays.append(self.da)
            self.fsm_state.append('DataArray')

        elif name == 'CoordinateSystemTransformMatrix':
            self.coordsys = gi.GiftiCoordSystem()
            self.img.darrays[-1].coordsys = self.coordsys
            self.fsm_state.append('CoordinateSystemTransformMatrix')

        elif name == 'DataSpace':
            if self.coordsys == None:
                raise ExpatError
            else:
                self.write_to = 'DataSpace'

        elif name == 'TransformedSpace':
            if self.coordsys == None:
                raise ExpatError
            else:
                self.write_to = 'TransformedSpace'

        elif name == 'MatrixData':
            if self.coordsys == None:
                raise ExpatError
            else:
                self.write_to = 'MatrixData'

        elif name == 'Data':
            self.write_to = 'Data'


    def EndElementHandler(self, name):
        if DEBUG_PRINT:
            print 'End element:\n\t', repr(name)

        if name == 'GIFTI':
            # remove last element of the list
            self.fsm_state.pop()
            # assert len(self.fsm_state) == 0

        elif name == 'MetaData':
            self.fsm_state.pop()
            
            if len(self.fsm_state) == 1:
                # only Gifti there, so this was a closing global
                # metadata tag
                self.img.meta = self.meta_global
                self.meta_global = None
            else:
                self.img.darrays[-1].meta = self.meta_da
                self.meta_da = None
                
        elif name == 'MD':
            self.fsm_state.pop()

            if not self.meta_global is None and self.meta_da == None:
                self.meta_global.data.append(self.nvpair)
            
            elif not self.meta_da is None and self.meta_global == None:
                self.meta_da.data.append(self.nvpair)
                
            # remove reference
            self.nvpair = None

        elif name == 'LabelTable':
            self.fsm_state.pop()

            # add labeltable
            self.img.labeltable = self.lata
            self.lata = None

        elif name == 'DataArray':
            if self.count_da:
                self.img.numDA += 1
            self.fsm_state.pop()
        elif name == 'CoordinateSystemTransformMatrix':
            self.fsm_state.pop()
            self.coordsys = None
        elif name == 'DataSpace':
            self.write_to = None
        elif name == 'TransformedSpace':
            self.write_to = None
        elif name == 'MatrixData':
            self.write_to = None
        elif name == 'Name':
            self.write_to = None
        elif name == 'Value':
            self.write_to = None
        elif name == 'Data':
            self.write_to = None
        elif name == 'Label':
            self.lata.labels.append(self.label)
            self.label = None
            self.write_to = None


    def CharacterDataHandler(self, data):

        if self.write_to == 'Name':
            data = data.strip()
            self.nvpair.name = data
        elif self.write_to == 'Value':
            data = data.strip()
            self.nvpair.value = data
        elif self.write_to == 'DataSpace':
            data = data.strip()
            self.coordsys.dataspace = xform_codes.code[data]
        elif self.write_to == 'TransformedSpace':
            data = data.strip()
            self.coordsys.xformspace = xform_codes.code[data]
        elif self.write_to == 'MatrixData':
            # conversion to numpy array
            c = StringIO(data)
            self.coordsys.xform = np.loadtxt(c)
            c.close()
        elif self.write_to == 'Data':
            da_tmp = self.img.darrays[-1]
            da_tmp.data = read_data_block(da_tmp.encoding, da_tmp.endian, \
                                          da_tmp.ind_ord, da_tmp.datatype, \
                                          da_tmp.dims, data)
        elif self.write_to == 'Label':
            self.label.label = data.strip()


def parse_gifti_file(fname, buffer_size = 35000000):

    datasource = open(fname,'rb')
    
    parser = ParserCreate()
    parser.buffer_text = True
    parser.buffer_size = buffer_size
    HANDLER_NAMES = [
    'StartElementHandler', 'EndElementHandler',
    'CharacterDataHandler',
    ]
    out = Outputter()
    for name in HANDLER_NAMES:
        setattr(parser, name, getattr(out, name))

    try:
        parser.ParseFile(datasource)
    except ExpatError:
        print 'An expat error occured while parsing the  Gifti file.'

    # update filename
    out.img.filename = fname
    return out.img
