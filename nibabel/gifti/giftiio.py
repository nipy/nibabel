
from parse_gifti_fast import *

##############
# General Gifti Input - Output to the filesystem
##############

def loadImage(filename):
    """ Load a Gifti image from a file """
    import os.path
    if not os.path.exists(filename):
        raise IOError("No such file or directory: '%s'" % filename)
    else:
        giifile = parse_gifti_file(filename)
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

    raise NotImplementedError("Writing Gifti Images is not implemented yet.")
