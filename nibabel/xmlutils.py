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
from xml.parsers.expat import ParserCreate


class XmlSerializable(object):
    """ Basic interface for serializing an object to xml"""

    def _to_xml_element(self):
        """ Output should be a xml.etree.ElementTree.Element"""
        raise NotImplementedError()

    def to_xml(self, enc='utf-8'):
        """ Output should be an xml string with the given encoding.
        (default: utf-8)"""
        ele = self._to_xml_element()
        if ele is None:
            return ''
        else:
            return tostring(ele, enc)


class XmlParser(object):
    """Thin wrapper around ParserCreate"""

    def __init__(self, encoding=None, buffer_size=3500000, verbose=0):
        self.parser = ParserCreate(encoding=encoding)
        self.parser.buffer_text = True
        self.parser.buffer_size = buffer_size
        self.verbose = verbose

    def parse(self, string):
        HANDLER_NAMES = ['StartElementHandler',
                         'EndElementHandler',
                         'CharacterDataHandler']
        for name in HANDLER_NAMES:
            setattr(self.parser, name, getattr(self, name))
        return self.parser.Parse(string)  # may throw ExpatError

    def parse_file(self, filename):
        with open(filename, 'r') as fp:
            return self.parse(fp.read())
