# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Read / write access to CaretSpecFile format

The format of CaretSpecFiles does not seem to have any independent
documentation.

Code can be found here [0], and a DTD was worked out in this email thread [1].

[0]: https://github.com/Washington-University/workbench/tree/master/src/Files
[1]: https://groups.google.com/a/humanconnectome.org/g/hcp-users/c/EGuwdaTVFuw/m/tg7a_-7mAQAJ
"""
import xml.etree.ElementTree as et
from urllib.parse import urlparse

import nibabel as nb
from nibabel import pointset as ps
from nibabel import xmlutils as xml
from nibabel.caret import CaretMetaData


class CaretSpecDataFile(xml.XmlSerializable):
    """DataFile

    * Attributes

        * Structure - A string from the BrainStructure list to identify
          what structure this element refers to (usually left cortex,
          right cortex, or cerebellum).
        * DataFileType - A string from the DataFileType list
        * Selected - A boolean

    * Child Elements: [NA]
    * Text Content: A URI
    * Parent Element - CaretSpecFile

    Attributes
    ----------
    structure : str
        Name of brain structure
    data_file_type : str
        Type of data file
    selected : bool
        Used for workbench internals
    uri : str
        URI of data file
    """

    def __init__(self, structure=None, data_file_type=None, selected=None, uri=None):
        super().__init__()
        self.structure = structure
        self.data_file_type = data_file_type
        self.selected = selected
        self.uri = uri

        if data_file_type == 'SURFACE':
            self.__class__ = SurfaceDataFile

    def _to_xml_element(self):
        data_file = xml.Element('DataFile')
        data_file.attrib['Structure'] = str(self.structure)
        data_file.attrib['DataFileType'] = str(self.data_file_type)
        data_file.attrib['Selected'] = 'true' if self.selected else 'false'
        data_file.text = self.uri
        return data_file

    def __repr__(self):
        return self.to_xml().decode()


class SurfaceDataFile(ps.TriangularMesh, CaretSpecDataFile):
    _gifti = None
    _coords = None
    _triangles = None

    def _get_gifti(self):
        if self._gifti is None:
            parts = urlparse(self.uri)
            if parts.scheme == 'file':
                self._gifti = nb.load(parts.path)
            elif parts.scheme == '':
                self._gifti = nb.load(self.uri)
            else:
                self._gifti = nb.GiftiImage.from_url(self.uri)
        return self._gifti

    def get_triangles(self, name=None):
        if self._triangles is None:
            gifti = self._get_gifti()
            self._triangles = gifti.agg_data('triangle')
        return self._triangles

    def get_coords(self, name=None):
        if self._coords is None:
            gifti = self._get_gifti()
            self._coords = gifti.agg_data('pointset')
        return self._coords


class CaretSpecFile(xml.XmlSerializable):
    """Class for CaretSpecFile XML documents

    These are used to identify related surfaces and volumes for use with CIFTI-2
    data files.
    """

    def __init__(self, metadata=None, data_files=(), version='1.0'):
        super().__init__()
        if metadata is not None:
            metadata = CaretMetaData(metadata)
        self.metadata = metadata
        self.data_files = list(data_files)
        self.version = version

    def _to_xml_element(self):
        caret_spec = xml.Element('CaretSpecFile')
        caret_spec.attrib['Version'] = str(self.version)
        if self.metadata is not None:
            caret_spec.append(self.metadata._to_xml_element())
        for data_file in self.data_files:
            caret_spec.append(data_file._to_xml_element())
        return caret_spec

    def to_xml(self, enc='UTF-8', **kwargs):
        ele = self._to_xml_element()
        et.indent(ele, '    ')
        return et.tostring(ele, enc, xml_declaration=True, short_empty_elements=False, **kwargs)

    def __eq__(self, other):
        return self.to_xml() == other.to_xml()

    @classmethod
    def from_filename(klass, fname, **kwargs):
        parser = CaretSpecParser(**kwargs)
        with open(fname, 'rb') as fobj:
            parser.parse(fptr=fobj)
        return parser.caret_spec


class CaretSpecParser(xml.XmlParser):
    def __init__(self, encoding=None, buffer_size=3500000, verbose=0):
        super().__init__(encoding=encoding, buffer_size=buffer_size, verbose=verbose)
        self.fsm_state = []
        self.struct_state = []

        self.caret_spec = None

        # where to write CDATA:
        self.write_to = None

        # Collecting char buffer fragments
        self._char_blocks = []

    def StartElementHandler(self, name, attrs):
        self.flush_chardata()
        if name == 'CaretSpecFile':
            self.caret_spec = CaretSpecFile(version=attrs['Version'])
        elif name == 'MetaData':
            self.caret_spec.metadata = CaretMetaData()
        elif name == 'MD':
            self.fsm_state.append('MD')
            self.struct_state.append(['', ''])
        elif name in ('Name', 'Value'):
            self.write_to = name
        elif name == 'DataFile':
            selected_map = {'true': True, 'false': False}
            data_file = CaretSpecDataFile(
                structure=attrs['Structure'],
                data_file_type=attrs['DataFileType'],
                selected=selected_map[attrs['Selected']],
            )
            self.caret_spec.data_files.append(data_file)
            self.struct_state.append(data_file)
            self.write_to = 'DataFile'

    def EndElementHandler(self, name):
        self.flush_chardata()
        if name == 'CaretSpecFile':
            ...
        elif name == 'MetaData':
            ...
        elif name == 'MD':
            key, value = self.struct_state.pop()
            self.caret_spec.metadata[key] = value
        elif name in ('Name', 'Value'):
            self.write_to = None
        elif name == 'DataFile':
            self.struct_state.pop()
            self.write_to = None

    def CharacterDataHandler(self, data):
        """Collect character data chunks pending collation

        The parser breaks the data up into chunks of size depending on the
        buffer_size of the parser.  A large bit of character data, with standard
        parser buffer_size (such as 8K) can easily span many calls to this
        function.  We thus collect the chunks and process them when we hit start
        or end tags.
        """
        if self._char_blocks is None:
            self._char_blocks = []
        self._char_blocks.append(data)

    def flush_chardata(self):
        """Collate and process collected character data"""
        if self._char_blocks is None:
            return

        # Just join the strings to get the data.  Maybe there are some memory
        # optimizations we could do by passing the list of strings to the
        # read_data_block function.
        data = ''.join(self._char_blocks)
        # Reset the char collector
        self._char_blocks = None
        # Process data
        if self.write_to == 'Name':
            data = data.strip()  # .decode('utf-8')
            pair = self.struct_state[-1]
            pair[0] = data

        elif self.write_to == 'Value':
            data = data.strip()  # .decode('utf-8')
            pair = self.struct_state[-1]
            pair[1] = data

        elif self.write_to == 'DataFile':
            data = data.strip()
            self.struct_state[-1].uri = data
