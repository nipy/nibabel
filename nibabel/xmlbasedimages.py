# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Thin layer around xml.etree.ElementTree, to abstract nibabel xml support.
"""
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.parsers.expat import ParserCreate, ExpatError

from .filebasedimages import FileBasedHeader, FileBasedImage


class XmlSerializable(object):
    """ Basic interface for serializing an object to xml"""

    def _to_xml_element(self):
        """ Output should be a xml.etree.ElementTree.Element"""
        raise NotImplementedError()

    def to_xml(self, enc='utf-8'):
        """ Output should be an xml string with the given encoding.
        (default: utf-8)"""
        return tostring(self._to_xml_element(), enc)


class XmlBasedHeader(FileBasedHeader, XmlSerializable):
    pass


class XmlImageParser(object):
    """ Parse XML image"""

    HANDLER_NAMES = ['StartElementHandler',
                     'EndElementHandler',
                     'CharacterDataHandler']

    def __init__(self, encoding=None):
        self.encoding = encoding
        self.img = None

    def parse(self, string=None, fname=None, fptr=None):
        """
        Parameters
        ----------
        string : str
            string containing xml document

        fname : str
            file name of an xml document.

        fptr : file pointer
            open file pointer to an xml document

        Returns
        -------
        img : XmlBasedImage
        """
        if int(fname is not None) + int(fptr is not None) + int(fname is not None) != 1:
            raise ValueError('Exactly one of fptr, fname, string must be specified.')

        if string is not None:
            fptr = StringIO(string)
        elif fname is not None:
            fptr = open(fname, 'r')

        parser = ParserCreate()  # from xml package
        for name in self.HANDLER_NAMES:
            setattr(parser, name, getattr(self, name))
        parser.ParseFile(fptr)

        return self.img

    def StartElementHandler(self, name, attrs):
        raise NotImplementedError

    def EndElementHandler(self, name):
        raise NotImplementedError

    def CharacterDataHandler(self, data):
        raise NotImplementedError


class XmlBasedImage(FileBasedImage, XmlSerializable):
    parser = XmlImageParser

    def to_file_map(self, file_map=None):
        """ Save the current image to the specified file_map

        Parameters
        ----------
        file_map : string

        Returns
        -------
        None
        """
        if file_map is None:
            file_map = self.file_map
        f = file_map['image'].get_prepare_fileobj('wb')
        f.write(self.to_xml())

    @classmethod
    def from_file_map(klass, file_map):
        """ Load a Gifti image from a file_map

        Parameters
        file_map : string

        Returns
        -------
        img : GiftiImage
            Returns a GiftiImage
         """
        return parser.parse(
            fptr=file_map['image'].get_prepare_fileobj('rb'))
