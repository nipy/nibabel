import numpy

#  The following classes are the wrappers as defined by the
#  struct definitions in gifti_io.h

class GiftiMetaData(object):
    
    length = int
    name = str
    value = str
    
    # added attribute
    # convert: Name - Value --> Key - Value
     #data = {}
    
    def __init__(self):
                
        # create a list of GiftiNVPairs objects containing the Name and the Value
        self.data = []
    
    
class GiftiNVPairs(object):
    
    #length = int
    name = str
    value = str

class GiftiLabelTable(object):
    
    legnth = int
    index = int
    label = str # ??

class GiftiCoordSystem(object):
    
    dataspace = str
    xformspace = str
    xform = None # numpy array

class GiftiDataArray(object):
    
    intent = int
    datatype = int
    ind_ord = int
    num_dim = int
    dims = [] # c_int*6),
    encoding = int
    endian = int
    ext_fname = str
    ext_offset = None # c_longlong XXX
    
    
    data = None
    coordsys = GiftiCoordSystem
    
    #numCS = int
    #nvals = None # longlong
    #nbyper = int
    #ex_atrs = GiftiNVPairs
    
    def __init__(self):
        
        self.dims = []
        self.meta = GiftiMetaData()

class GiftiImage(object):
    
    numDA = int
    version = str
    filename = str
    
    #meta = GiftiMetaData
    
    # list of GiftiDataArray
    #darrays = []
        
    
    
    #darray = GiftiDataArray
    #swapped = int
    #ex_atrs = GiftiNVPairs
    
    def __init__(self):
        
        self.darrays = []
        self.meta = GiftiMetaData()
        #self.labeltable = GiftiLabelTable()
        

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
        # XXX sanity checks
        
    def remove_gifti_data_array(self, dataarr):
        self.darrays.remove(dataarr)
        # XXX update
    
    
#class GiftiGlobals(object):
#    
#    verb = int
#
#class GiftiTypeEle(object):
#    
#    type = int
#    nbyper = int
#    swapsize = int
#    name = str


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
		Generic GIFTI File    .gii
		Coordinates           .coord.gii
		Functional            .func.gii
		Labels                .label.gii
		RGB or RGBA           .rgba.gii
		Shape                 .shape.gii
		Surface               .surf.gii
		Tensors               .tensor.gii
		Time Series           .time.gii
		Topology              .topo.gii	"""
        
	#if not image.version:
	#	t = pygiftiio.gifticlib_version()
	#	versionstr = t[t.find("version ")+8:t.find(", ")]
	#	float(versionstr) # raise an exception should the format change in the future :-)
	#	image.version = versionstr
        
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
