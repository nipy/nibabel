# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from xml.parsers.expat import ParserCreate, ExpatError
from gifti import *
from util import *
from numpy import loadtxt
import numpy

parser = None
img = None
out = None

def read_data_block(encoding, endian, ordering, datatype, shape, data):
    """ Tries to unzip, decode, parse the funny string data """

    # XXX: how to incorporate endianness?

    import base64
    import zlib
    from StringIO import StringIO

    if ordering == 1:
        ord = 'C'
    elif ordering == 2:
        ord = 'F'
    else:
        ord = 'C'

    if encoding == 1:
        # GIFTI_ENCODING_ASCII
        c = StringIO(data)
        da = numpy.loadtxt(c)
        return da

    elif encoding == 2:
        # GIFTI_ENCODING_B64BIN
        dec = base64.decodestring(data)
        dt = GiftiType2npyType[datatype]
        sh = tuple(shape)
        return numpy.fromstring(zdec, dtype = dt).reshape(sh, order = ord)

    elif encoding == 3:
        # GIFTI_ENCODING_B64GZ
        dec = base64.decodestring(data)
        zdec = zlib.decompress(dec)
        dt = GiftiType2npyType[datatype]
        sh = tuple(shape)
        return numpy.fromstring(zdec, dtype = dt).reshape(sh, order = ord)

    elif encoding == 4:
        # GIFTI_ENCODING_EXTBIN
        # XXX: to be implemented. In what format are the external files?
        pass
    else:
        return 0

class Outputter(object):

    # finite state machine stack
    fsm_state = []

    # temporary constructs
    nvpair = None
    da = None
    coordsys = None
    lata = None
    label = None

    # where to write CDATA:
    write_to = None


    def StartElementHandler(self, name, attrs):
        #print 'Start element:\n\t', repr(name), attrs
        global img

        if name == 'GIFTI':
            # create gifti image
            img = GiftiImage()
            if attrs.has_key('Version'):
                img.version = attrs['Version']

            if attrs.has_key('NumberOfDataArrays'):
                img.numDA = attrs['NumberOfDataArrays']

            self.fsm_state.append('GIFTI')


        elif name == 'MetaData':

            self.fsm_state.append('MetaData')

        elif name == 'MD':

            self.nvpair = GiftiNVPairs()

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

            self.lata = GiftiLabelTable()

            self.fsm_state.append('LabelTable')

        elif name == 'Label':

            self.label = GiftiLabel()

            if attrs.has_key("Index"):
                self.label.index = int(attrs["Index"])

            self.write_to = 'Label'

        elif name == 'DataArray':

            self.da = GiftiDataArray()

            if attrs.has_key("Intent"):
                self.da.intent = GiftiIntentCode.intents[attrs["Intent"]]

            if attrs.has_key("DataType"):
                self.da.datatype = GiftiDataType.datatypes[attrs["DataType"]]

            if attrs.has_key("ArrayIndexingOrder"):
                self.da.ind_ord = GiftiArrayIndexOrder.ordering[attrs["ArrayIndexingOrder"]]

            if attrs.has_key("Dimensionality"):
                self.da.num_dim = int(attrs["Dimensionality"])

            for i in range(self.da.num_dim):
                di = "Dim%s" % str(i)
                if attrs.has_key(di):
                    self.da.dims.append(int(attrs[di]))

            # dimensionality has to correspond to the number of DimX given
            assert len(self.da.dims) == self.da.num_dim

            if attrs.has_key("Encoding"):
                self.da.encoding = GiftiEncoding.encodings[attrs["Encoding"]]

            if attrs.has_key("Endian"):
                self.da.endian = GiftiEndian.endian[attrs["Endian"]]

            if attrs.has_key("ExternalFileName"):
                self.da.ext_fname = attrs["ExternalFileName"]

            if attrs.has_key("ExternalFileOffset"):
                self.da.ext_offset = attrs["ExternalFileOffset"]

            img.darrays.append(self.da)

            self.fsm_state.append('DataArray')

        elif name == 'CoordinateSystemTransformMatrix':

            self.coordsys = GiftiCoordSystem()

            img.darrays[-1].coordsys = self.coordsys

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
        #print 'End element:\n\t', repr(name)

        global img
        if name == 'GIFTI':
            # remove last element of the list
            self.fsm_state.pop()
            # assert len(self.fsm_state) == 0
            print self.fsm_state

        elif name == 'MetaData':
            self.fsm_state.pop()
        elif name == 'MD':
            self.fsm_state.pop()

            # add nvpair to correct metadata

            # case for either Gifti MetaData or DataArray Metadata
            if self.fsm_state[1] == 'MetaData':
                # Gifti MetaData
                img.meta.data.append(self.nvpair)
            elif self.fsm_state[1] == 'DataArray' and self.fsm_state[2] == 'MetaData':
                # append to last DataArray
                img.darrays[-1].meta.data.append(self.nvpair)

            # remove reference
            self.nvpair = None

        elif name == 'LabelTable':
            self.fsm_state.pop()

            # add labeltable
            img.labeltable = self.lata
            self.lata = None

        elif name == 'DataArray':
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
            self.coordsys.dataspace = data
        elif self.write_to == 'TransformedSpace':
            data = data.strip()
            self.coordsys.xformspace = data
        elif self.write_to == 'MatrixData':
            # conversion to numpy array
            from StringIO import StringIO
            c = StringIO(data)
            self.coordsys.xform = loadtxt(c)
            c.close()
        elif self.write_to == 'Data':
            da_tmp = img.darrays[-1]
            da_tmp.data = read_data_block(da_tmp.encoding, da_tmp.endian, \
                                          da_tmp.ind_ord, da_tmp.datatype, \
                                          da_tmp.dims, data)
        elif self.write_to == 'Label':
            self.label.label = data


def parse_gifti_file(fname):

    datasource = open(fname,'r')
    global img
    global parser
    global out

    img = None
    out = None
    parser = None

    parser = ParserCreate()
    parser.buffer_text = True
    parser.buffer_size = 35000000
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
        print 'An error occured while parsing Gifti file.'

    # update filename
    img.filename = fname

    return img
