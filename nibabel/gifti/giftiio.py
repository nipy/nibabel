# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
# General Gifti Input - Output to and from the filesystem
# Stephan Gerhard, Oktober 2010
##############

import os
import codecs

from .parse_gifti_fast import parse_gifti_file

def read(filename):
    """ Load a Gifti image from a file

    Parameters
    ----------
    filename : string
        The Gifti file to open, it has usually ending .gii

    Returns
    -------
    img : GiftiImage
        Returns a GiftiImage
     """
    if not os.path.isfile(filename):
        raise IOError("No such file or directory: '%s'" % filename)
    return parse_gifti_file(filename)


def write(image, filename):
    """ Save the current image to a new file

    Parameters
    ----------
    image : GiftiImage
        A GiftiImage instance to store 
    filename : string
        Filename to store the Gifti file to

    Returns
    -------
    None

    Notes
    -----
    We write all files with utf-8 encoding, and specify this at the top of the
    XML file with the ``encoding`` attribute.

    The Gifti spec suggests using the following suffixes to your
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

    The Gifti file is stored in endian convention of the current machine.
    """
    # Our giftis are always utf-8 encoded - see GiftiImage.to_xml
    with codecs.open(filename, 'wb', encoding='utf-8') as f:
        f.write(image.to_xml())
