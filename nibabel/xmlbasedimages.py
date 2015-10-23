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


class XmlBasedImage(FileBasedImage, XmlSerializable):

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
