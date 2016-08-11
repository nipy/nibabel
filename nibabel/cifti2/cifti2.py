# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Read / write access to CIfTI2 image format

Format of the NIFTI2 container format described here:

    http://www.nitrc.org/forum/message.php?msg_id=3738

Definition of the CIFTI2 header format and file extensions here:

    https://www.nitrc.org/forum/attachment.php?attachid=333&group_id=454&forum_id=1955

'''
from __future__ import division, print_function, absolute_import
import re
import collections

import numpy as np

from .. import xmlutils as xml
from ..filebasedimages import FileBasedHeader, FileBasedImage
from ..nifti2 import Nifti2Image


class CIFTI2HeaderError(Exception):
    """ Error in CIFTI2 header
    """


CIFTI_MAP_TYPES = ('CIFTI_INDEX_TYPE_BRAIN_MODELS',
                   'CIFTI_INDEX_TYPE_PARCELS',
                   'CIFTI_INDEX_TYPE_SERIES',
                   'CIFTI_INDEX_TYPE_SCALARS',
                   'CIFTI_INDEX_TYPE_LABELS')

CIFTI_MODEL_TYPES = ('CIFTI_MODEL_TYPE_SURFACE',
                     'CIFTI_MODEL_TYPE_VOXELS')

CIFTI_SERIESUNIT_TYPES = ('SECOND',
                          'HERTZ',
                          'METER',
                          'RADIAN')

CIFTI_BrainStructures = ('CIFTI_STRUCTURE_ACCUMBENS_LEFT',
                         'CIFTI_STRUCTURE_ACCUMBENS_RIGHT',
                         'CIFTI_STRUCTURE_ALL_WHITE_MATTER',
                         'CIFTI_STRUCTURE_ALL_GREY_MATTER',
                         'CIFTI_STRUCTURE_AMYGDALA_LEFT',
                         'CIFTI_STRUCTURE_AMYGDALA_RIGHT',
                         'CIFTI_STRUCTURE_BRAIN_STEM',
                         'CIFTI_STRUCTURE_CAUDATE_LEFT',
                         'CIFTI_STRUCTURE_CAUDATE_RIGHT',
                         'CIFTI_STRUCTURE_CEREBELLAR_WHITE_MATTER_LEFT',
                         'CIFTI_STRUCTURE_CEREBELLAR_WHITE_MATTER_RIGHT',
                         'CIFTI_STRUCTURE_CEREBELLUM',
                         'CIFTI_STRUCTURE_CEREBELLUM_LEFT',
                         'CIFTI_STRUCTURE_CEREBELLUM_RIGHT',
                         'CIFTI_STRUCTURE_CEREBRAL_WHITE_MATTER_LEFT',
                         'CIFTI_STRUCTURE_CEREBRAL_WHITE_MATTER_RIGHT',
                         'CIFTI_STRUCTURE_CORTEX',
                         'CIFTI_STRUCTURE_CORTEX_LEFT',
                         'CIFTI_STRUCTURE_CORTEX_RIGHT',
                         'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT',
                         'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT',
                         'CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT',
                         'CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT',
                         'CIFTI_STRUCTURE_OTHER',
                         'CIFTI_STRUCTURE_OTHER_GREY_MATTER',
                         'CIFTI_STRUCTURE_OTHER_WHITE_MATTER',
                         'CIFTI_STRUCTURE_PALLIDUM_LEFT',
                         'CIFTI_STRUCTURE_PALLIDUM_RIGHT',
                         'CIFTI_STRUCTURE_PUTAMEN_LEFT',
                         'CIFTI_STRUCTURE_PUTAMEN_RIGHT',
                         'CIFTI_STRUCTURE_THALAMUS_LEFT',
                         'CIFTI_STRUCTURE_THALAMUS_RIGHT')


def _value_if_klass(val, klass, check_isinstance_or_none=True):
    if check_isinstance_or_none and val is None:
        return val
    elif isinstance(val, klass):
        return val
    raise ValueError('Not a valid %s instance.' % klass.__name__)


def _underscore(string):
    """ Convert a string from CamelCase to underscored """
    string = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', string)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', string).lower()


class Cifti2MetaData(xml.XmlSerializable, collections.MutableMapping):
    """ A list of name-value pairs

    Attributes
    ----------
    data : list of (name, value) tuples
    """
    def __init__(self):
        self.data = collections.OrderedDict()

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def difference_update(self, metadata):
        """Remove metadata key-value pairs

        Parameters
        ----------
        metadata : dict-like datatype

        Returns
        -------
        None

        """
        if metadata is None:
            raise ValueError("The metadata parameter can't be None")
        pairs = dict(metadata)
        for k in pairs:
            del self.data[k]

    def _to_xml_element(self):
        metadata = xml.Element('MetaData')

        for name_text, value_text in self.data.items():
            md = xml.SubElement(metadata, 'MD')
            name = xml.SubElement(md, 'Name')
            name.text = str(name_text)
            value = xml.SubElement(md, 'Value')
            value.text = str(value_text)
        return metadata


class Cifti2LabelTable(xml.XmlSerializable, collections.MutableMapping):
    """ Cifti2 label table: a sequence of ``Cifti2Label``s
    """

    def __init__(self):
        self._labels = collections.OrderedDict()

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, key):
        return self._labels[key]

    def append(self, label):
        self[label.key] = label

    def __setitem__(self, key, value):
        if isinstance(value, Cifti2Label):
            if key != value.key:
                raise ValueError("The key and the label's key must agree")
            self._labels[key] = value
        else:
            try:
                key = int(key)
                v = (str(value[0]),) + tuple(float(v) for v in value[1:] if 0 <= float(v) <= 1)
                if len(v) != 5:
                    raise ValueError
            except:
                raise ValueError(
                    'Key must be integer and value a string and 4-tuple of floats between 0 and 1'
                )

            label = Cifti2Label(
                key=key,
                label=v[0],
                red=v[1],
                green=v[2],
                blue=v[3],
                alpha=v[4]
            )
            self._labels[key] = label

    def __delitem__(self, key):
        del self._labels[key]

    def __iter__(self):
        return iter(self._labels)

    def _to_xml_element(self):
        if len(self) == 0:
            raise CIFTI2HeaderError('LabelTable element requires at least 1 label')
        labeltable = xml.Element('LabelTable')
        for ele in self._labels.values():
            labeltable.append(ele._to_xml_element())
        return labeltable


class Cifti2Label(xml.XmlSerializable):
    """ Cifti2 label: association of integer key with a name and RGBA values

    Attribute descriptions are from the CIFTI-2 spec dated 2014-03-01.
    For all color components, value is floating point with range 0.0 to 1.0.

    Attributes
    ----------
    key : int
        Integer, data value which is assigned this name and color.
    label : str
        Name of the label.
    red : None or float
        Red color component for label.
    green : None or float
        Green color component for label.
    blue : None or float
        Blue color component for label.
    alpha : None or float
        Alpha color component for label.
    """
    def __init__(self, key=0, label='', red=0, green=0, blue=0,
                 alpha=0):
        self.key = key
        self.label = label
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha

    @property
    def rgba(self):
        """ Returns RGBA as tuple """
        return (self.red, self.green, self.blue, self.alpha)

    def _to_xml_element(self):
        if self.label is '':
            raise CIFTI2HeaderError('Label needs a name')
        try:
            v = int(self.key)
        except ValueError:
            raise CIFTI2HeaderError('The key must be an integer')
        for c_ in ('red', 'blue', 'green', 'alpha'):
            try:
                v = float(getattr(self, c_))
                if not (0 <= v <= 1):
                    raise ValueError
            except ValueError:
                raise CIFTI2HeaderError(
                    'Label invalid %s needs to be a float between 0 and 1. and it is %s' %
                    (c_, v)
                )

        lab = xml.Element('Label')
        lab.attrib['Key'] = str(self.key)
        lab.text = str(self.label)

        for name in ('red', 'green', 'blue', 'alpha'):
            attr = str(getattr(self, name))
            lab.attrib[name.capitalize()] = attr
        return lab


class Cifti2NamedMap(xml.XmlSerializable):
    """Cifti2 named map: association of name and optional data with a map index

    Associates a name, optional metadata, and possibly a LabelTable with an
    index in a map.

    Attributes
    ----------
    map_name : str
        Name of map
    metadata : None or Cifti2MetaData
        Metadata associated with named map
    label_table : None or Cifti2LabelTable
        Label table associated with named map
    """
    def __init__(self, map_name=None, metadata=None, label_table=None):
        self.map_name = map_name
        self.metadata = metadata
        self.label_table = label_table

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        """ Set the metadata for this NamedMap

        Parameters
        ----------
        meta : Cifti2MetaData

        Returns
        -------
        None
        """
        self._metadata = _value_if_klass(metadata, Cifti2MetaData)

    @property
    def label_table(self):
        return self._label_table

    @label_table.setter
    def label_table(self, label_table):
        """ Set the label_table for this NamedMap

        Parameters
        ----------
        label_table : Cifti2LabelTable

        Returns
        -------
        None
        """
        self._label_table = _value_if_klass(label_table, Cifti2LabelTable)

    def _to_xml_element(self):
        named_map = xml.Element('NamedMap')
        if self.metadata:
            named_map.append(self.metadata._to_xml_element())
        if self.label_table:
            named_map.append(self.label_table._to_xml_element())
        map_name = xml.SubElement(named_map, 'MapName')
        map_name.text = self.map_name
        return named_map


class Cifti2Surface(xml.XmlSerializable):
    """Cifti surface: association of brain structure and number of vertices

    "Specifies the number of vertices for a surface, when IndicesMapToDataType
    is 'CIFTI_INDEX_TYPE_PARCELS.' This is separate from the Parcel element
    because there can be multiple parcels on one surface, and one parcel may
    involve multiple surfaces."

    Attributes
    ----------
    brain_structure : str
        Name of brain structure
    surface_number_of_vertices : int
        Number of vertices on surface
    """
    def __init__(self, brain_structure=None, surface_number_of_vertices=None):
        self.brain_structure = brain_structure
        self.surface_number_of_vertices = surface_number_of_vertices

    def _to_xml_element(self):
        if self.brain_structure is None:
            raise CIFTI2HeaderError('Surface element requires at least 1 BrainStructure')
        surf = xml.Element('Surface')
        surf.attrib['BrainStructure'] = str(self.brain_structure)
        surf.attrib['SurfaceNumberOfVertices'] = str(self.surface_number_of_vertices)
        return surf


class Cifti2VoxelIndicesIJK(xml.XmlSerializable, collections.MutableSequence):
    """Cifti2 VoxelIndicesIJK: Set of voxel indices contained in a structure

    "Identifies the voxels that model a brain structure, or participate in a
    parcel. Note that when this is a child of BrainModel, the IndexCount
    attribute of the BrainModel indicates the number of voxels contained in
    this element."

    Each element of this sequence is a triple of integers.
    """
    def __init__(self, indices=None):
        self._indices = []
        if indices is not None:
            self.extend(indices)

    def __len__(self):
        return len(self._indices)

    def __delitem__(self, index):
        if not isinstance(index, int) and len(index) > 1:
            raise NotImplementedError
        del self._indices[index]

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._indices[index]
        elif len(index) == 2:
            if not isinstance(index[0], int):
                raise NotImplementedError
            return self._indices[index[0]][index[1]]
        else:
            raise ValueError('Only row and row,column access is allowed')

    def __setitem__(self, index, value):
        if isinstance(index, int):
            try:
                value = [int(v) for v in value]
                if len(value) != 3:
                    raise ValueError('rows are triples of ints')
                self._indices[index[0]] = value
            except ValueError:
                raise ValueError('value must be a triple of ints')
        elif len(index) == 2:
            try:
                if not isinstance(index[0], int):
                    raise NotImplementedError
                value = int(value)
                self._indices[index[0]][index[1]] = value
            except ValueError:
                raise ValueError('value must be an int')
        else:
            raise ValueError

    def insert(self, index, value):
        if not isinstance(index, int) and len(index) != 1:
            raise ValueError('Only rows can be inserted')
        try:
            value = [int(v) for v in value]
            if len(value) != 3:
                raise ValueError
            self._indices.insert(index, value)
        except ValueError:
            raise ValueError('value must be a triple of int')

    def _to_xml_element(self):
        if len(self) == 0:
            raise CIFTI2HeaderError('VoxelIndicesIJK element require an index table')

        vox_ind = xml.Element('VoxelIndicesIJK')
        vox_ind.text = '\n'.join(' '.join([str(v) for v in row])
                                 for row in self._indices)
        return vox_ind


class Cifti2Vertices(xml.XmlSerializable):
    """Cifti2 vertices - association of brain structure and a list of vertices

    "Contains a BrainStructure type and a list of vertex indices within a
    Parcel."

    Attribute descriptions are from the CIFTI-2 spec dated 2014-03-01.

    Attributes
    ----------
    brain_structure : str
        A string from the BrainStructure list to identify what surface this
        vertex list is from (usually left cortex, right cortex, or cerebellum).
    vertices : ndarray shape (N,)
        Vertex indices (which are independent for each surface, and zero-based)
    """
    def __init__(self, brain_structure=None, vertices=None):
        self.vertices = vertices
        self.brain_structure = brain_structure

    def _to_xml_element(self):
        if self.brain_structure is None:
            raise CIFTI2HeaderError('Vertices element require a BrainStructure')

        vertices = xml.Element('Vertices')
        vertices.attrib['BrainStructure'] = str(self.brain_structure)

        if self.vertices is not None:
            vertices.text = ' '.join(self.vertices.astype(str))
        return vertices


class Cifti2Parcel(xml.XmlSerializable):
    """Cifti2 parcel: association of a name with vertices and/or voxels

    Attributes
    ----------
    name : str
        Name of parcel
    voxel_indices_ijk : None or Cifti2VoxelIndicesIJK
        Voxel indices associated with parcel
    vertices : list of Cifti2Vertices
        Vertices associated with parcel
    """
    def __init__(self, name=None, voxel_indices_ijk=None, vertices=None):
        self.name = name
        self.voxel_indices_ijk = voxel_indices_ijk
        self.vertices = vertices if vertices is not None else []

    @property
    def voxel_indices_ijk(self):
        return self._voxel_indices_ijk

    @voxel_indices_ijk.setter
    def voxel_indices_ijk(self, value):
        self._voxel_indices_ijk = _value_if_klass(value, Cifti2VoxelIndicesIJK)

    def add_cifti_vertices(self, vertices):
        """ Adds a vertices to the Cifti2Parcel

        Parameters
        ----------
        vertices : Cifti2Vertices
        """
        if not isinstance(vertices, Cifti2Vertices):
            raise TypeError("Not a valid Cifti2Vertices instance")
        self.vertices.append(vertices)

    def remove_cifti2_vertices(self, ith):
        """ Removes the ith vertices element from the Cifti2Parcel """
        self.vertices.pop(ith)

    def _to_xml_element(self):
        if self.name is None:
            raise CIFTI2HeaderError('Parcel element requires a name')

        parcel = xml.Element('Parcel')
        parcel.attrib['Name'] = str(self.name)
        if self.voxel_indices_ijk:
            parcel.append(self.voxel_indices_ijk._to_xml_element())
        for vertex in self.vertices:
            parcel.append(vertex._to_xml_element())
        return parcel


class Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(xml.XmlSerializable):
    """Matrix that translates voxel indices to spatial coordinates

    Attributes
    ----------
    meter_exponent : int
        "[S]pecifies that the coordinate result from the transformation matrix
        should be multiplied by 10 to this power to get the spatial coordinates
        in meters (e.g., if this is '-3', then the transformation matrix is in
        millimeters)."
    matrix : array-like shape (4, 4)
        Affine transformation matrix from voxel indices to RAS space
    """
    # meterExponent = int
    # matrix = np.array

    def __init__(self, meter_exponent=None, matrix=None):
        self.meter_exponent = meter_exponent
        self.matrix = matrix

    def _to_xml_element(self):
        if self.matrix is None:
            raise CIFTI2HeaderError(
                'TransformationMatrixVoxelIndicesIJKtoXYZ element requires a matrix'
            )
        trans = xml.Element('TransformationMatrixVoxelIndicesIJKtoXYZ')
        trans.attrib['MeterExponent'] = str(self.meter_exponent)
        trans.text = '\n'.join(' '.join(map('{:.10f}'.format, row))
                               for row in self.matrix)
        return trans


class Cifti2Volume(xml.XmlSerializable):
    """Cifti2 volume: information about a volume for mappings that use voxels

    Attributes
    ----------
    volume_dimensions : array-like shape (3,)
        "[T]he lengthss of the three volume file dimensions that are related to
        spatial coordinates, in number of voxels. Voxel indices (which are
        zero-based) that are used in the mapping that this element applies to
        must be within these dimensions."
    transformation_matrix_voxel_indices_ijk_to_xyz \
        : Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ
        Matrix that translates voxel indices to spatial coordinates
    """
    def __init__(self, volume_dimensions=None, transform_matrix=None):
        self.volume_dimensions = volume_dimensions
        self.transformation_matrix_voxel_indices_ijk_to_xyz = transform_matrix

    def _to_xml_element(self):
        if self.volume_dimensions is None:
            raise CIFTI2HeaderError('Volume element requires dimensions')

        volume = xml.Element('Volume')
        volume.attrib['VolumeDimensions'] = ','.join(
            [str(val) for val in self.volume_dimensions])
        volume.append(self.transformation_matrix_voxel_indices_ijk_to_xyz._to_xml_element())
        return volume


class Cifti2VertexIndices(xml.XmlSerializable, collections.MutableSequence):
    """Cifti2 vertex indices: vertex indices for an associated brain model

    The vertex indices (which are independent for each surface, and
    zero-based) that are used in this brain model[.] The parent
    BrainModel's ``index_count`` indicates the number of indices.
    """
    def __init__(self, indices=None):
        self._indices = []
        if indices is not None:
            self.extend(indices)

    def __len__(self):
        return len(self._indices)

    def __delitem__(self, index):
        del self._indices[index]

    def __getitem__(self, index):
        return self._indices[index]

    def __setitem__(self, index, value):
        try:
            value = int(value)
            self._indices[index] = value
        except ValueError:
            raise ValueError('value must be an int')

    def insert(self, index, value):
        try:
            value = int(value)
            self._indices.insert(index, value)
        except ValueError:
            raise ValueError('value must be an int')

    def _to_xml_element(self):
        if len(self) == 0:
            raise CIFTI2HeaderError('VertexIndices element requires indices')

        vert_indices = xml.Element('VertexIndices')
        vert_indices.text = ' '.join([str(i) for i in self])
        return vert_indices


class Cifti2BrainModel(xml.XmlSerializable):

    # index_offset = int
    # index_count = int
    # model_type = str
    # brain_structure = str
    # surface_number_of_vertices = int
    # voxel_indices_ijk = np.array
    # vertex_indices = np.array

    def __init__(self, index_offset=None, index_count=None, model_type=None,
                 brain_structure=None, n_surface_vertices=None,
                 voxel_indices_ijk=None, vertex_indices=None):
        self.index_offset = index_offset
        self.index_count = index_count
        self.model_type = model_type
        self.brain_structure = brain_structure
        self.surface_number_of_vertices = n_surface_vertices

        self.voxel_indices_ijk = voxel_indices_ijk
        self.vertex_indices = vertex_indices

    @property
    def voxel_indices_ijk(self):
        return self._voxel_indices_ijk

    @voxel_indices_ijk.setter
    def voxel_indices_ijk(self, value):
        self._voxel_indices_ijk = _value_if_klass(value, Cifti2VoxelIndicesIJK)

    @property
    def vertex_indices(self):
        return self._vertex_indices

    @vertex_indices.setter
    def vertex_indices(self, value):
        self._vertex_indices = _value_if_klass(value, Cifti2VertexIndices)

    def _to_xml_element(self):
        brain_model = xml.Element('BrainModel')

        for key in ['IndexOffset', 'IndexCount', 'ModelType', 'BrainStructure',
                    'SurfaceNumberOfVertices']:
            attr = _underscore(key)
            value = getattr(self, attr)
            if value is not None:
                brain_model.attrib[key] = str(value)
        if self.voxel_indices_ijk:
            brain_model.append(self.voxel_indices_ijk._to_xml_element())
        if self.vertex_indices:
            brain_model.append(self.vertex_indices._to_xml_element())
        return brain_model


class Cifti2MatrixIndicesMap(xml.XmlSerializable, collections.MutableSequence):
    """Class for Matrix Indices Map

    Provides a mapping between matrix indices and their interpretation.
    """
    # applies_to_matrix_dimension = list
    # indices_map_to_data_type = str
    # number_of_series_points = int
    # series_exponent = int
    # series_start = float
    # series_step = float
    # series_unit = str
    _valid_type_mappings_ = {
        Cifti2BrainModel: ('CIFTI_INDEX_TYPE_BRAIN_MODELS',),
        Cifti2Parcel: ('CIFTI_INDEX_TYPE_PARCELS',),
        Cifti2NamedMap: ('CIFTI_INDEX_TYPE_LABELS',),
        Cifti2Volume: ('CIFTI_INDEX_TYPE_SCALARS', 'CIFTI_INDEX_TYPE_SERIES'),
        Cifti2Surface: ('CIFTI_INDEX_TYPE_SCALARS', 'CIFTI_INDEX_TYPE_SERIES')
    }

    def __init__(self, applies_to_matrix_dimension,
                 indices_map_to_data_type,
                 number_of_series_points=None,
                 series_exponent=None,
                 series_start=None,
                 series_step=None,
                 series_unit=None,
                 maps=[],
                 ):
        self.applies_to_matrix_dimension = applies_to_matrix_dimension
        self.indices_map_to_data_type = indices_map_to_data_type
        self.number_of_series_points = number_of_series_points
        self.series_exponent = series_exponent
        self.series_start = series_start
        self.series_step = series_step
        self.series_unit = series_unit
        self._maps = []
        for m in maps:
            self.append(m)

    def __len__(self):
        return len(self._maps)

    def __delitem__(self, index):
        del self._maps[index]

    def __getitem__(self, index):
        return self._maps[index]

    def __setitem__(self, index, value):
        if (
            isinstance(value, Cifti2Volume) and
            (
                self.volume is not None and
                not isinstance(self._maps[index], Cifti2Volume)
            )
        ):
            raise CIFTI2HeaderError("Only one Volume can be in a MatrixIndicesMap")
        self._maps[index] = value

    def insert(self, index, value):
        if (
            isinstance(value, Cifti2Volume) and
            self.volume is not None
        ):
            raise CIFTI2HeaderError("Only one Volume can be in a MatrixIndicesMap")

        self._maps.insert(index, value)

    @property
    def named_maps(self):
        for p in self:
            if isinstance(p, Cifti2NamedMap):
                yield p

    @property
    def surfaces(self):
        for p in self:
            if isinstance(p, Cifti2Surface):
                yield p

    @property
    def parcels(self):
        for p in self:
            if isinstance(p, Cifti2Parcel):
                yield p

    @property
    def volume(self):
        for p in self:
            if isinstance(p, Cifti2Volume):
                return p
        return None

    @volume.setter
    def volume(self, volume):
        if not isinstance(volume, Cifti2Volume):
            raise ValueError("You can only set a volume with a volume")
        for i, v in enumerate(self):
            if isinstance(v, Cifti2Volume):
                break
        else:
            self.append(volume)
            return
        self[i] = volume

    @volume.deleter
    def volume(self):
        for i, v in enumerate(self):
            if isinstance(v, Cifti2Volume):
                break
        else:
            raise ValueError("No Cifti2Volume element")
        del self[i]

    @property
    def brain_models(self):
        for p in self:
            if isinstance(p, Cifti2BrainModel):
                yield p

    def _to_xml_element(self):
        if self.applies_to_matrix_dimension is None:
            raise CIFTI2HeaderError(
                'MatrixIndicesMap element requires to be applied to at least 1 dimension'
            )

        mat_ind_map = xml.Element('MatrixIndicesMap')
        dims_as_strings = [str(dim) for dim in self.applies_to_matrix_dimension]
        mat_ind_map.attrib['AppliesToMatrixDimension'] = ','.join(dims_as_strings)
        for key in ['IndicesMapToDataType', 'NumberOfSeriesPoints', 'SeriesExponent',
                    'SeriesStart', 'SeriesStep', 'SeriesUnit']:
            attr = _underscore(key)
            value = getattr(self, attr)
            if value is not None:
                mat_ind_map.attrib[key] = str(value)
        for map_ in self:
            mat_ind_map.append(map_._to_xml_element())

        return mat_ind_map


class Cifti2Matrix(xml.XmlSerializable, collections.MutableSequence):
    def __init__(self):
        self._mims = []
        self.metadata = None

    @property
    def metadata(self):
        return self._meta

    @metadata.setter
    def metadata(self, meta):
        """ Set the metadata for this Cifti2Header

        Parameters
        ----------
        meta : Cifti2MetaData

        Returns
        -------
        None
        """
        if meta is not None and not isinstance(meta, Cifti2MetaData):
            raise TypeError("Not a valid Cifti2MetaData instance")
        self._meta = meta

    def __setitem__(self, key, value):
        if not isinstance(mim, Cifti2MatrixIndicesMap):
            raise TypeError("Not a valid Cifti2MatrixIndicesMap instance")
        self._mims[key] = value

    def __getitem__(self, key):
        return self._mims[key]

    def __delitem__(self, key):
        del self._mims[key]

    def __len__(self):
        return len(self._mims)

    def insert(self, index, value):
        if not isinstance(value, Cifti2MatrixIndicesMap):
            raise TypeError("Not a valid Cifti2MatrixIndicesMap instance")
        self._mims.insert(index, value)

    def _to_xml_element(self):
        if (len(self) == 0 and self.metadata is None):
            raise CIFTI2HeaderError(
                'Matrix element requires either a MatrixIndicesMap or a Metadata element'
            )

        mat = xml.Element('Matrix')
        if self.metadata:
            mat.append(self.metadata._to_xml_element())
        for mim in self._mims:
            mat.append(mim._to_xml_element())
        return mat


class Cifti2Header(FileBasedHeader, xml.XmlSerializable):
    ''' Class for Cifti2 header extension '''

    def __init__(self, matrix=None, version="2.0"):
        FileBasedHeader.__init__(self)
        xml.XmlSerializable.__init__(self)
        self.matrix = Cifti2Matrix() if matrix is None else Cifti2Matrix()
        self.version = version

    def _to_xml_element(self):
        cifti = xml.Element('CIFTI')
        cifti.attrib['Version'] = str(self.version)
        mat_xml = self.matrix._to_xml_element()
        if mat_xml is not None:
            cifti.append(mat_xml)
        return cifti

    @classmethod
    def from_header(klass, header=None):
        if header is None:
            return klass()
        if type(header) == klass:
            return header.copy()
        raise ValueError('header is not a Cifti2Header')

    @classmethod
    def may_contain_header(klass, binaryblock):
        from .parse_cifti2_fast import _Cifti2AsNiftiHeader
        return _Cifti2AsNiftiHeader.may_contain_header(binaryblock)


class Cifti2Image(FileBasedImage):
    """ Class for single file CIfTI2 format image
    """
    header_class = Cifti2Header
    valid_exts = Nifti2Image.valid_exts
    files_types = Nifti2Image.files_types
    makeable = False
    rw = True

    def __init__(self, data=None, header=None, nifti_header=None):
        ''' Initialize image

        The image is a combination of (array, affine matrix, header, nifti_header),
        with optional metadata in `extra`, and filename / file-like objects
        contained in the `file_map` mapping.

        Parameters
        ----------
        dataobj : object
           Object containg image data.  It should be some object that retuns an
           array from ``np.asanyarray``.  It should have a ``shape`` attribute
           or property
        affine : None or (4,4) array-like
           homogenous affine giving relationship between voxel coordinates and
           world coordinates.  Affine can also be None.  In this case,
           ``obj.affine`` also returns None, and the affine as written to disk
           will depend on the file format.
        header : Cifti2Header object
        nifti_header : None or mapping or nifti2 header instance, optional
           metadata for this image format
        '''
        self._header = header or Cifti2Header()
        self.data = data
        self.extra = nifti_header

    def get_data(self):
        return self.data

    @property
    def shape(self):
        return self.data.shape

    @classmethod
    def from_file_map(klass, file_map):
        """ Load a Cifti2 image from a file_map

        Parameters
        file_map : string

        Returns
        -------
        img : Cifti2Image
            Returns a Cifti2Image
         """
        from .parse_cifti2_fast import _Cifti2AsNiftiImage, Cifti2Extension
        nifti_img = _Cifti2AsNiftiImage.from_file_map(file_map)

        # Get cifti2 header
        for item in nifti_img.header.extensions:
            if isinstance(item, Cifti2Extension):
                cifti_header = item.get_content()
                break
        else:
            cifti_header = None

        if cifti_header is None:
            raise ValueError('Nifti2 header does not contain a CIFTI2 '
                             'extension')

        # Construct cifti image
        cifti_img = Cifti2Image(data=nifti_img.get_data()[0, 0, 0, 0],
                                header=cifti_header,
                                nifti_header=nifti_img.header)
        cifti_img.file_map = nifti_img.file_map
        return cifti_img

    @classmethod
    def from_image(klass, img):
        ''' Class method to create new instance of own class from `img`

        Parameters
        ----------
        img : ``spatialimage`` instance
           In fact, an object with the API of ``FileBasedImage``.

        Returns
        -------
        cimg : ``spatialimage`` instance
           Image, of our own class
        '''
        if isinstance(img, klass):
            return img
        else:
            raise NotImplementedError

    def to_file_map(self, file_map=None):
        """ Save the current image to the specified file_map

        Parameters
        ----------
        file_map : string

        Returns
        -------
        None
        """
        from .parse_cifti2_fast import Cifti2Extension
        header = self.extra
        extension = Cifti2Extension(content=self.header.to_xml())
        header.extensions.append(extension)
        data = np.reshape(self.data, (1, 1, 1, 1) + self.data.shape)
        # If qform not set, reset pixdim values so Nifti2 does not complain
        if header['qform_code'] == 0:
            header['pixdim'][:4] = 1
        img = Nifti2Image(data, None, header)
        img.to_file_map(file_map or self.file_map)


class Cifti2DenseDataSeriesHeader(Cifti2Header):
    @classmethod
    def may_contain_header(klass, binaryblock):
        from .parse_cifti2_fast import _Cifti2DenseDataSeriesNiftiHeader
        return _Cifti2DenseDataSeriesNiftiHeader.may_contain_header(binaryblock)


class Cifti2DenseDataSeries(Cifti2Image):
    """Class to handle Dense Data Series
    Dense Data Series
    -----------------

    Intent_code: 3002, NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES
    Intent_name: ConnDenseSeries
    File extension: .dtseries.nii
    AppliesToMatrixDimension 0: series
    AppliesToMatrixDimension 1: brain models

    This file type represents data points in a series for every vertex and voxel
    in the mapping.  A row is a complete data series,for a single vertex or
    voxel in the mapping that applies along the second dimension. A data series
    is often a timeseries, but it can also represent other data types such as a
    series of sampling depths along the surface normal from the white to pial
    surface.  It retains the 't' in dtseries from CIFTI-1 naming conventions.
    """
    header_class = Cifti2DenseDataSeriesHeader
    valid_exts = ('.dtseries.nii',)
    files_types = (('image', '.dtseries.nii'),)


def load(filename):
    """ Load cifti2 from `filename`

    Parameters
    ----------
    filename : str
        filename of image to be loaded

    Returns
    -------
    img : Cifti2Image
        cifti image instance

    Raises
    ------
    ImageFileError: if `filename` doesn't look like cifti
    IOError : if `filename` does not exist
    """
    return Cifti2Image.from_filename(filename)


def save(img, filename):
    """ Save cifti to `filename`

    Parameters
    ----------
    filename : str
        filename to which to save image
    """
    Cifti2Image.instance_to_filename(img, filename)
