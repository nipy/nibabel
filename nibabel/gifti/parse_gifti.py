""" Exploration of Python capabilities for parsing Gifti XML """

from xml.dom.minidom import parse, parseString
from gifti import GiftiImage, GiftiMetaData, GiftiLabelTable, GiftiDataArray
from util import GiftiIntentCode, GiftiDataType, GiftiEncoding, GiftiEndian, \
                GiftiArrayIndexOrder, GiftiType2npyType
import numpy

def parse_metadata(dom_node):
    """ Parse metadata information and return a dictionary """
    
    meta = GiftiMetaData()
    
    MD = dom_node.getElementsByTagName("MD")
    for e3 in MD:
        if e3.nodeType == e3.ELEMENT_NODE and e3.localName == "MD":
            
            name = ''
            value = ''
            # only keep the last occuring Name and Value values
            for e4 in e3.getElementsByTagName("Name"):
                name = e4.childNodes[0].data
            for e4 in e3.getElementsByTagName("Value"):
                value = e4.childNodes[0].data
                
            if not name == '':
                if not value == '':
                    meta.data[name] = value
    return meta

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
    

def parse_dataarray(dom_node):
    """ Parse GiftiDataArray and return it """
    
    da = GiftiDataArray()
    
    if dom_node.hasAttribute("Intent"):
        da.intent = GiftiIntentCode.intents[dom_node.getAttribute("Intent")]

    if dom_node.hasAttribute("DataType"):
        da.datatype = GiftiDataType.datatypes[dom_node.getAttribute("DataType")]
        
    if dom_node.hasAttribute("ArrayIndexingOrder"):
        da.ind_ord = GiftiArrayIndexOrder.ordering[dom_node.getAttribute("ArrayIndexingOrder")]

    if dom_node.hasAttribute("Dimensionality"):
        da.num_dim = int(dom_node.getAttribute("Dimensionality"))

    for i in range(da.num_dim):
        di = "Dim%s" % str(i)
        if dom_node.hasAttribute(di):
            da.dims.append(int(dom_node.getAttribute(di)))

    # dimensionality has to correspond to the number of DimX given
    assert len(da.dims) == da.num_dim

    if dom_node.hasAttribute("Encoding"):
        da.encoding = GiftiEncoding.encodings[dom_node.getAttribute("Encoding")]

    if dom_node.hasAttribute("Endian"):
        da.endian = GiftiEndian.endian[dom_node.getAttribute("Endian")]
    
    if dom_node.hasAttribute("ExternalFileName"):
        da.ext_fname = dom_node.getAttribute("ExternalFileName")
        
    if dom_node.hasAttribute("ExternalFileOffset"):
        da.ext_offset = dom_node.getAttribute("ExternalFileOffset")

    meta = parse_metadata(dom_node)
    da.meta = meta
    
    DAT = dom_node.getElementsByTagName("Data")

    if not len(DAT) == 0:

        # take only the first node
        data_node = DAT[0]
        
        data = data_node.childNodes[0].data
        
        # XXX sanity check if all required attributes are set
        
        da.data = read_data_block(da.encoding, da.endian, da.ind_ord, da.datatype, da.dims, data)
            
    return da

    
def parse_gifti_file(fname):

    datasource = open(fname)
    dom = parse(datasource)   # parse an open file

    img = None

    if dom.hasChildNodes():
        
        for e in dom.childNodes:
            if e.nodeType == e.ELEMENT_NODE and e.localName == "GIFTI":
                # Gifti instance
                
                img = GiftiImage()
                
                if e.hasAttribute("Version"):
                    img.version = e.getAttribute("Version")
                    
                if e.hasAttribute("NumberOfDataArrays"):
                    img.numDA = int(e.getAttribute("NumberOfDataArrays"))
                    
                for e2 in e.childNodes:
                    
                    if e2.nodeType == e2.TEXT_NODE:
                        continue
                    
                    elif e2.nodeType == e.ELEMENT_NODE and e2.localName == "MetaData":
                        
                        meta = parse_metadata(e2)
                        # setting GiftiMetaData
                        img.set_metadata(meta)
     
                    elif e2.nodeType == e.ELEMENT_NODE and e2.localName == "LabelTable":
                        
                        labeltable = GiftiLabelTable()
                        # parse it
                        
                    elif e2.nodeType == e.ELEMENT_NODE and e2.localName == "DataArray":
                        
                    
                        dataarray = parse_dataarray(e2)
                        
                        # add dataarray to GiftiImage
                        img.add_gifti_data_array(dataarray)
        
    return img
    
    