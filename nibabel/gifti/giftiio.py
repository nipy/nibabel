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

from ..deprecated import deprecate_with_version


@deprecate_with_version('giftiio.read function deprecated. '
                        "Use nibabel.load() instead.",
                        '2.1', '4.0')
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
    from ..loadsave import load
    return load(filename)


@deprecate_with_version('giftiio.write function deprecated. '
                        "Use nibabel.load() instead.",
                        '2.1', '4.0')
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
    from ..loadsave import save
    return save(image, filename)
