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

from io import BytesIO
from xml.etree.ElementTree import Element, SubElement, tostring  # flake8: noqa aliasing
from xml.parsers.expat import ParserCreate

from .filebasedimages import FileBasedHeader


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
    """ Basic wrapper around FileBasedHeader and XmlSerializable."""


class XmlParser(object):
    """ Base class for defining how to parse xml-based image snippets.

    Image-specific parsers should define:
        StartElementHandler
        EndElementHandler
        CharacterDataHandler
    """

    HANDLER_NAMES = ['StartElementHandler',
                     'EndElementHandler',
                     'CharacterDataHandler']

    def __init__(self, encoding=None, buffer_size=35000000, verbose=0):
        """
        Parameters
        ----------
        encoding : str
            string containing xml document

        buffer_size: None or int, optional
            size of read buffer. None uses default buffer_size
            from xml.parsers.expat.

        verbose : int, optional
            amount of output during parsing (0=silent, by default).
        """
        self.encoding = encoding
        self.buffer_size = buffer_size
        self.verbose = verbose

    def _create_parser(self):
        """Internal function that allows subclasses to mess
        with the underlying parser, if desired."""

        parser = ParserCreate(encoding=self.encoding)  # from xml package
        parser.buffer_text = True
        if self.buffer_size is not None:
            parser.buffer_size = self.buffer_size
        return parser

    def parse(self, string=None, fname=None, fptr=None):
        """
        Parameters
        ----------
        string : str
            string containing xml document

        fname : str
            file name of an xml document.

        fptr : file pointer
            open file pointer to an xml documents
        """
        if int(string is not None) + int(fptr is not None) + int(fname is not None) != 1:
            raise ValueError('Exactly one of fptr, fname, string must be specified.')

        if string is not None:
            fptr = BytesIO(string)
        elif fname is not None:
            fptr = open(fname, 'r')

        parser = self._create_parser()
        for name in self.HANDLER_NAMES:
            setattr(parser, name, getattr(self, name))
        parser.ParseFile(fptr)

    def StartElementHandler(self, name, attrs):
        raise NotImplementedError

    def EndElementHandler(self, name):
        raise NotImplementedError

    def CharacterDataHandler(self, data):
        raise NotImplementedError
