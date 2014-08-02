# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
''' Read / write access to NIfTI2 image format

Format described here:

    http://www.nitrc.org/forum/message.php?msg_id=3738

Stuff about the CIFTI file format here:

    http://www.nitrc.org/plugins/mwiki/index.php/cifti:ConnectivityMatrixFileFormats

'''
from __future__ import division, print_function, absolute_import

from StringIO import StringIO
from xml.parsers.expat import ParserCreate, ExpatError

import numpy as np

DEBUG_PRINT = False

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


class CiftiMetaData(object):
    """ A list of GiftiNVPairs in stored in
    the list self.data """
    def __init__(self, nvpair = None):
        self.data = []
        if not nvpair is None:
            if isinstance(nvpair, list):
                self.data.extend(nvpair)
            else:
                self.data.append(nvpair)

    @classmethod
    def from_dict(klass, data_dict):
        meda = klass()
        for k,v in data_dict.items():
            nv = CiftiNVPair(k, v)
            meda.data.append(nv)
        return meda

    def get_metadata(self):
        """ Returns metadata as dictionary """
        self.data_as_dict = {}
        for ele in self.data:
            self.data_as_dict[ele.name] = ele.value
        return self.data_as_dict

    def to_xml(self, prefix='', indent='    '):
        if len(self.data) == 0:
            return ''
        res = "%s<MetaData>\n" % prefix
        preindent = prefix + indent
        for ele in self.data:
            nvpair = """%s<MD>
%s<Name>%s</Name>
%s<Value>%s</Value>
%s</MD>\n""" % (preindent, preindent + indent, ele.name, preindent + indent,
                  ele.value, preindent)
            res += nvpair
        res += "%s</MetaData>\n" % prefix
        return res

    def print_summary(self):
        print(self.get_metadata())


class CiftiNVPair(object):

    name = str
    value = str

    def __init__(self, name = '', value = ''):
        self.name = name
        self.value = value

class CiftiLabelTable(object):

    def __init__(self):
        self.labels = []

    def get_labels_as_dict(self):
        self.labels_as_dict = {}
        for ele in self.labels:
            self.labels_as_dict[ele.key] = ele.label
        return self.labels_as_dict

    def to_xml(self, prefix='', indent='    '):
        if len(self.labels) == 0:
            return ''
        res = "%s<LabelTable>\n" % prefix
        for ele in self.labels:
            col = ''
            if not ele.red is None:
                col += ' Red="%s"' % str(ele.red)
            if not ele.green is None:
                col += ' Green="%s"' % str(ele.green)
            if not ele.blue is None:
                col += ' Blue="%s"' % str(ele.blue)
            if not ele.alpha is None:
                col += ' Alpha="%s"' % str(ele.alpha)
            lab = """%s<Label Key="%s"%s><![CDATA[%s]]></Label>\n""" % \
                (prefix + indent, str(ele.key), col, ele.label)
            res += lab
        res += "%s</LabelTable>\n" % prefix
        return res

    def print_summary(self):
        print(self.get_labels_as_dict())


class CiftiLabel(object):
    key = int
    label = str
    # rgba
    # freesurfer examples seem not to conform
    # to datatype "NIFTI_TYPE_RGBA32" because they
    # are floats, not unsigned 32-bit integers
    red = float
    green = float
    blue = float
    alpha = float

    def __init__(self, key = 0, label = '', red = None,\
                  green = None, blue = None, alpha = None):
        self.key = key
        self.label = label
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha

    def get_rgba(self):
        """ Returns RGBA as tuple """
        return (self.red, self.green, self.blue, self.alpha)


class CiftiNamedMap(object):
    """Class for Named Map"""
    map_name = str

    def __init__(self, map_name=None, meta=None, label_table=None):
        self.map_name = map_name
        if meta is None:
            self.meta = CiftiMetaData()
        else:
            assert isinstance(meta, CiftiMetaData)
            self.meta = meta
        if label_table is None:
            self.label_table = CiftiLabelTable()
        else:
            assert isinstance(meta, CiftiLabelTable)
            self.label_table = label_table

    def get_metadata(self):
        return self.meta

    def set_metadata(self, meta):
        """ Set the metadata for this NamedMap

        Parameters
        ----------
        meta : CiftiMetaData

        Returns
        -------
        None
        """
        if isinstance(meta, CiftiMetaData):
            self.meta = meta
            print("New Metadata set.")
        else:
            print("Not a valid CiftiMetaData instance")

    def get_label_table(self):
        return self.label_table

    def set_label_table(self, label_table):
        """ Set the label_table for this NamedMap

        Parameters
        ----------
        label_table : CiftiLabelTable

        Returns
        -------
        None
        """
        if isinstance(label_table, CiftiLabelTable):
            self.label_table = label_table
            print("New LabelTable set.")
        else:
            print("Not a valid CiftiLabelTable instance")

    def to_xml(self, prefix='', indent='    '):
        if self.map_name is None:
            return ''
        res = "%s<NamedMap>\n" % prefix
        res += self.meta.to_xml(prefix=prefix + indent, indent=indent)
        res += self.label_table.to_xml(prefix=prefix + indent, indent=indent)
        res += "%s<MapName>%s</MapName>\n" % (prefix + indent, self.map_name)
        res += "%s</NamedMap>\n" % prefix
        return res


class CiftiSurface(object):
    """Class for Surface """
    brainStructure = str
    surfaceNumberOfVertices = int

    def __init__(self, brainStructure=None, surfaceNumberOfVertices=None):
        self.brainStructure = brainStructure
        self.surfaceNumberOfVertices = surfaceNumberOfVertices

    def to_xml(self, prefix='', indent='    '):
        if self.brainStructure is None:
            return ''
        res = ('%s<Surface BrainStructure="%s" '
               'SurfaceNumberOfVertices"%s" />\n') % (prefix,
                                                      self.brainStructure,
                                                      self.surfaceNumberOfVertices)
        return res

class CiftiVoxelIndicesIJK(object):
    indices = np.array

    def __init__(self, indices=None):
        self.indices = indices

    def to_xml(self, prefix='', indent='    '):
        if self.indices is None:
            return ''
        res = '%s<VoxelIndicesIJK>' % prefix
        for row in self.indices:
            res += ' '.join(row.astype(str).tolist()) + '\n'
        res += '</VoxelIndicesIJK>\n'
        return res


class CiftiVertices(object):

    brainStructure = str
    vertices = np.array

    def __init__(self, brain_structure=None, vertices=None):
        self.vertices = vertices
        self.brainStructure = brain_structure

    def to_xml(self, prefix='', indent='    '):
        if self.vertices is None:
            return ''
        res = '%s<Vertices BrainStructure="%s">' % (prefix, self.brainStructure)
        res += ' '.join(self.vertices.astype(str).tolist())
        res += '</Vertices>\n'
        return res


class CiftiParcel(object):
    """Class for Parcel"""
    name = str
    numVA = int

    def __init__(self, name=None, voxel_indices_ijk=None, vertices=None):
        self.name = name
        self.voxelIndicesIJK = voxel_indices_ijk
        if voxel_indices_ijk is not None:
            self._voxelIndicesIJK = voxel_indices_ijk
        if vertices is None:
            vertices = []
        self.vertices = vertices
        self.numVA = len(vertices)

    @property
    def voxelIndicesIJK(self):
        return self._voxelIndicesIJK

    @voxelIndicesIJK.setter
    def voxelIndicesIJK(self, value):
        assert isinstance(value, CiftiVoxelIndicesIJK)
        self._voxelIndicesIJK = value

    def add_cifti_vertices(self, vertices):
        """ Adds a vertices to the CiftiParcel

        Parameters
        ----------
        vertices : CiftiVertices
        """
        if isinstance(vertices, CiftiVertices):
            self.vertices.append(vertices)
            self.numVA += 1
        else:
            print("mim paramater must be of type CiftiMatrixIndicesMap")

    def remove_cifti_vertices(self, ith):
        """ Removes the ith vertices element from the CiftiParcel """
        self.vertices.pop(ith)
        self.numVA -= 1

    def to_xml(self, prefix='', indent='    '):
        if self.name is None:
            return ''
        res = '%s<Parcel Name="%s">\n' % (prefix, self.name)
        res += self.voxelIndicesIJK.to_xml(prefix=prefix + indent, indent=indent)
        for vertices in self.vertices:
            res += vertices.to_xml(prefix=prefix + indent, indent=indent)
        res += "%s</Parcel>\n" % prefix
        return res


class CiftiTransformationMatrixVoxelIndicesIJKtoXYZ(object):

    meterExponent = int
    matrix = np.array

    def __init__(self, meter_exponent=None, matrix=None):
        self.meterExponent = meter_exponent
        self.matrix = matrix

    def to_xml(self, prefix='', indent='    '):
        if self.matrix is None:
            return ''
        res = ('%s<TransformationMatrixVoxelIndices'
               'IJKtoXYZ MeterExponend="%d">') % (prefix, self.meterExponent)
        for row in self.matrix:
            res += '\n' + ' '.join(['%.10f' % val for val in row])
        res += "</TransformationMatrixVoxelIndicesIJKtoXYZ>\n"
        return res


class CiftiVolume(object):

    volumeDimensions = np.array
    transformationMatrixVoxelIndicesIJKtoXYZ = np.array

    def __init__(self, volume_dimensions=None, transform_matrix=None):
        self.volumeDimensions = volume_dimensions
        self.transformationMatrixVoxelIndicesIJKtoXYZ = transform_matrix

    def to_xml(self, prefix='', indent='    '):
        if not self.volumeDimensions:
            return ''
        res = '%s<Volume VolumeDimensions="%s">\n' % (prefix,
              ','.join([str(val) for val in self.volumeDimensions]))
        res += self.transformationMatrixVoxelIndicesIJKtoXYZ.to_xml(prefix=prefix + '\t')
        res += "%s</Volume>\n" % prefix
        return res


class CiftiVertexIndices(object):
    indices = np.array

    def __init__(self, indices=None):
        self.indices = indices

    def to_xml(self, prefix='', indent='    '):
        if self.indices is None:
            return ''
        indices = ' '.join(self.indices.astype(str).tolist())
        res = '%s<VertexIndices>%s</VertexIndices>\n' % (prefix, indices)
        return res


class CiftiBrainModel(object):

    indexOffset = int
    indexCount = int
    modelType = str
    brainStructure = str
    surfaceNumberOfVertices = int
    _voxelIndicesIJK = np.array
    _vertexIndices = np.array

    def __init__(self, index_offset=None, index_count=None, model_type=None,
                 brain_structure=None, n_surface_vertices=None,
                 voxel_indices_ijk=None, vertex_indices=None):
        self.indexOffset = index_offset
        self.indexCount = index_count
        self.modelType = model_type
        self.brainStructure = brain_structure
        self.surfaceNumberOfVertices = n_surface_vertices

        if voxel_indices_ijk is not None:
            self.voxelIndicesIJK = voxel_indices_ijk
        else:
            self.voxelIndicesIJK = CiftiVoxelIndicesIJK()
        if vertex_indices is not None:
            self.vertexIndices = vertex_indices
        else:
            self.vertexIndices = CiftiVertexIndices()

    @property
    def voxelIndicesIJK(self):
        return self._voxelIndicesIJK

    @voxelIndicesIJK.setter
    def voxelIndicesIJK(self, value):
        assert isinstance(value, CiftiVoxelIndicesIJK)
        self._voxelIndicesIJK = value

    @property
    def vertexIndices(self):
        return self._vertexIndices

    @vertexIndices.setter
    def vertexIndices(self, value):
        assert isinstance(value, CiftiVertexIndices)
        self._vertexIndices = value

    def to_xml(self, prefix='', indent='    '):
        if self.indexOffset is None:
            return ''
        attrs = []
        for key in ['IndexOffset', 'IndexCount', 'ModelType', 'BrainStructure',
                    'SurfaceNumberOfVertices']:
            attr = key[0].lower() + key[1:]
            value = getattr(self, attr)
            if value is not None:
                attrs += ['%s="%s"' % (key, value)]
        attrs = ' '.join(attrs)
        res = '%s<BrainModel %s>\n' % (prefix, attrs)
        if self.voxelIndicesIJK:
            res += self.voxelIndicesIJK.to_xml(prefix=prefix + indent,
                                               indent=indent)
        if self.vertexIndices:
            res += self.vertexIndices.to_xml(prefix=prefix + indent,
                                             indent=indent)
        res += "%s</BrainModel>\n" % prefix
        return res


class CiftiMatrixIndicesMap(object):
    """Class for Matrix Indices Map

    Provides a mapping between matrix indices and their interpretation.
    """
    numBrainModels = int
    numNamedMaps = int
    numParcels = int
    numSurfaces = int
    appliesToMatrixDimension = int
    indicesMapToDataType = str
    numberOfSeriesPoints = int
    seriesExponent = int
    seriesStart = float
    seriesStep = float
    seriesUnit = str

    def __init__(self, appliesToMatrixDimension,
                 indicesMapToDataType,
                 numberOfSeriesPoints=None,
                 seriesExponent=None,
                 seriesStart=None,
                 seriesStep=None,
                 seriesUnit=None,
                 brainModels=None,
                 namedMaps=None,
                 parcels=None,
                 surfaces=None,
                 volume=None):
        self.appliesToMatrixDimension = appliesToMatrixDimension
        self.indicesMapToDataType = indicesMapToDataType
        self.numberOfSeriesPoints = numberOfSeriesPoints
        self.seriesExponent = seriesExponent
        self.seriesStart = seriesStart
        self.seriesStep = seriesStep
        self.seriesUnit = seriesUnit
        if brainModels is None:
            brainModels = []
        self.brainModels = brainModels
        self.numBrainModels = len(self.brainModels)
        if namedMaps is None:
            namedMaps = []
        self.namedMaps = namedMaps
        self.numNamedMaps = len(self.namedMaps)
        if parcels is None:
            parcels = []
        self.parcels = parcels
        self.numParcels = len(self.parcels)
        if surfaces is None:
            surfaces = []
        self.surfaces = surfaces
        self.numSurfaces = len(self.surfaces)
        self.volume = CiftiVolume()
        if not volume and isinstance(volume, CiftiVolume):
            self.volume = volume

    def add_cifti_brain_model(self, brain_model):
        """ Adds a brain model to the CiftiMatrixIndicesMap

        Parameters
        ----------
        brain_model : CiftiBrainModel
        """
        if isinstance(brain_model, CiftiBrainModel):
            self.brainModels.append(brain_model)
            self.numBrainModels += 1
        else:
            print("brain_model parameter must be of type CiftiBrainModel")

    def remove_cifti_brain_model(self, ith):
        """ Removes the ith brain model element from the CiftiMatrixIndicesMap """
        self.brainModels.pop(ith)
        self.numBrainModels -= 1

    def add_cifti_named_map(self, named_map):
        """ Adds a named map to the CiftiMatrixIndicesMap

        Parameters
        ----------
        named_map : CiftiNamedMap
        """
        if isinstance(named_map, CiftiNamedMap):
            self.namedMaps.append(named_map)
            self.numNamedMaps += 1
        else:
            print("named_map parameter must be of type CiftiNamedMap")

    def remove_cifti_named_map(self, ith):
        """ Removes the ith named_map element from the CiftiMatrixIndicesMap """
        self.namedMaps.pop(ith)
        self.numNamedMaps -= 1

    def add_cifti_parcel(self, parcel):
        """ Adds a parcel to the CiftiMatrixIndicesMap

        Parameters
        ----------
        parcel : CiftiParcel
        """
        if isinstance(parcel, CiftiParcel):
            self.parcels.append(parcel)
            self.numParcels += 1
        else:
            print("parcel parameter must be of type CiftiParcel")

    def remove_cifti_parcel(self, ith):
        """ Removes the ith parcel element from the CiftiMatrixIndicesMap """
        self.parcels.pop(ith)
        self.numParcels -= 1

    def add_cifti_surface(self, surface):
        """ Adds a surface to the CiftiMatrixIndicesMap

        Parameters
        ----------
        surface : CiftiSurface
        """
        if isinstance(surface, CiftiSurface):
            self.surfaces.append(surface)
            self.numSurfaces += 1
        else:
            print("surface parameter must be of type CiftiSurface")

    def remove_cifti_surface(self, ith):
        """ Removes the ith surface element from the CiftiMatrixIndicesMap """
        self.surfaces.pop(ith)
        self.numSurfaces -= 1

    def set_cifti_volume(self, volume):
        """ Adds a volume to the CiftiMatrixIndicesMap

        Parameters
        ----------
        volume : CiftiVolume
        """
        if isinstance(volume, CiftiVolume):
            self.volume = volume
        else:
            print("volume parameter must be of type CiftiVolume")

    def remove_cifti_volume(self):
        """ Removes the volume element from the CiftiMatrixIndicesMap """
        self.volume = None

    def to_xml(self, prefix='', indent='    '):
        if self.appliesToMatrixDimension is None:
            return ''
        attrs = []
        for key in ['AppliesToMatrixDimension', 'IndicesMapToDataType',
                    'NumberOfSeriesPoints', 'SeriesExponent', 'SeriesStart',
                    'SeriesStep', 'SeriesUnit']:
            attr = key[0].lower() + key[1:]
            value = getattr(self, attr)
            if value is not None:
                attrs += ['%s="%s"' % (key, value)]
        attrs = ' '.join(attrs)
        res = '%s<MatrixIndicesMap %s>\n' % (prefix, attrs)
        for named_map in self.namedMaps:
            res += named_map.to_xml(prefix=prefix + indent, indent=indent)
        for surface in self.surfaces:
            res += surface.to_xml(prefix=prefix + indent, indent=indent)
        for parcel in self.parcels:
            res += parcel.to_xml(prefix=prefix + indent, indent=indent)
        if self.volume:
            res += self.volume.to_xml(prefix=prefix + indent, indent=indent)
        for model in self.brainModels:
            res += model.to_xml(prefix=prefix + indent, indent=indent)
        res += "%s</MatrixIndicesMap>\n" % prefix
        return res


class CiftiMatrix(object):

    numMIM = int

    def __init__(self, meta=None, mims=None):
        if mims is None:
            mims = []
        self.mims = mims
        if meta is None:
            self.meta = CiftiMetaData()
        else:
            self.meta = meta
        self.numMIM = len(self.mims)

    def get_metadata(self):
        return self.meta

    def set_metadata(self, meta):
        """ Set the metadata for this CiftiHeader

        Parameters
        ----------
        meta : CiftiMetaData

        Returns
        -------
        None
        """
        if isinstance(meta, CiftiMetaData):
            self.meta = meta
            print("New Metadata set.")
        else:
            print("Not a valid CiftiMetaData instance")

    def add_cifti_matrix_indices_map(self, mim):
        """ Adds a matrix indices map to the CiftiMatrix

        Parameters
        ----------
        mim : CiftiMatrixIndicesMap
        """
        if isinstance(mim, CiftiMatrixIndicesMap):
            self.mims.append(mim)
            self.numMIM += 1
        else:
            print("mim paramater must be of type CiftiMatrixIndicesMap")

    def remove_cifti_matrix_indices_map(self, ith):
        """ Removes the ith matrix indices map element from the CiftiMatrix """
        self.mims.pop(ith)
        self.numMIM -= 1

    def to_xml(self, prefix='', indent='    '):
        if self.numMIM == 0:
            return ''
        res = '%s<Matrix>\n' % prefix
        if self.meta:
            res += self.meta.to_xml(prefix=prefix + indent, indent=indent)
        for mim in self.mims:
            res += mim.to_xml(prefix=prefix + indent, indent=indent)
        res += "%s</Matrix>\n" % prefix
        return res


class CiftiHeader(object):
    ''' Class for Cifti2 header extension '''

    version = str

    def __init__(self, matrix=None, version="2.0"):
        if matrix is None:
            self.matrix = CiftiMatrix()
        else:
            self.matrix = matrix
        self.version = version

    def to_xml(self, prefix='', indent='    '):
        res = '%s<CIFTI Version="%s">\n' % (prefix, self.version)
        res += self.matrix.to_xml(prefix=prefix + indent, indent=indent)
        res += "%s</CIFTI>\n" % prefix
        return res


class Outputter(object):

    def __init__(self, intent_code, filename):
        self.intent_code = intent_code
        self.data_type = '.'.join(filename.split('.')[:-2])
        self.initialize()

    def initialize(self):
        """ Initialize outputter
        """
        # finite state machine stack
        self.fsm_state = []
        self.struct_state = []

        # temporary constructs
        self.nvpair = None
        self.da = None
        self.coordsys = None
        self.lata = None
        self.label = None

        self.meta = None
        self.count_mim = True

        # where to write CDATA:
        self.write_to = None
        self.header = None

        # Collecting char buffer fragments
        self._char_blocks = None

    def StartElementHandler(self, name, attrs):
        self.flush_chardata()
        if DEBUG_PRINT:
            print('Start element:\n\t', repr(name), attrs)
        if name == 'CIFTI':
            # create gifti image
            self.header = CiftiHeader()
            if 'Version' in attrs:
                self.header.version = attrs['Version']
            self.fsm_state.append('CIFTI')
            self.struct_state.append(self.header)
        elif name == 'Matrix':
            self.fsm_state.append('Matrix')
            matrix = CiftiMatrix()
            header = self.struct_state[-1]
            assert isinstance(header, CiftiHeader)
            header.matrix = matrix
            self.struct_state.append(matrix)
        elif name == 'MetaData':
            self.fsm_state.append('MetaData')
            meta = CiftiMetaData()
            parent = self.struct_state[-1]
            assert isinstance(parent, (CiftiMatrix, CiftiNamedMap))
            self.struct_state.append(meta)
        elif name == 'MD':
            pair = CiftiNVPair()
            self.fsm_state.append('MD')
            self.struct_state.append(pair)
        elif name == 'Name':
            self.write_to = 'Name'
        elif name == 'Value':
            self.write_to = 'Value'
        elif name == 'MatrixIndicesMap':
            self.fsm_state.append('MatrixIndicesMap')
            mim = CiftiMatrixIndicesMap(appliesToMatrixDimension=int(attrs["AppliesToMatrixDimension"]),
                                        indicesMapToDataType=attrs["IndicesMapToDataType"])
            for key, dtype in [("NumberOfSeriesPoints", int),
                               ("SeriesExponent", int),
                               ("SeriesStart", float),
                               ("SeriesStep", float),
                               ("SeriesUnit", str)]:
                if key in attrs:
                    var = key[0].lower() + key[1:]
                    setattr(mim, var, dtype(attrs[key]))
            matrix = self.struct_state[-1]
            assert isinstance(matrix, CiftiMatrix)
            matrix.add_cifti_matrix_indices_map(mim)
            self.struct_state.append(mim)
        elif name == 'NamedMap':
            self.fsm_state.append('NamedMap')
            named_map = CiftiNamedMap()
            mim = self.struct_state[-1]
            assert isinstance(mim, CiftiMatrixIndicesMap)
            self.struct_state.append(named_map)
            mim.add_cifti_named_map(named_map)
        elif name == 'LabelTable':
            named_map = self.struct_state[-1]
            mim = self.struct_state[-2]
            assert mim.indicesMapToDataType == "CIFTI_INDEX_TYPE_LABELS"
            lata = CiftiLabelTable()
            assert isinstance(named_map, CiftiNamedMap)
            self.fsm_state.append('LabelTable')
            self.struct_state.append(lata)
        elif name == 'Label':
            lata = self.struct_state[-1]
            assert isinstance(lata, CiftiLabelTable)
            label = CiftiLabel()
            if "Key" in attrs:
                label.key = int(attrs["Key"])
            if "Red" in attrs:
                label.red = float(attrs["Red"])
            if "Green" in attrs:
                label.green = float(attrs["Green"])
            if "Blue" in attrs:
                label.blue = float(attrs["Blue"])
            if "Alpha" in attrs:
                label.alpha = float(attrs["Alpha"])
            self.write_to = 'Label'
            self.fsm_state.append('Label')
            self.struct_state.append(label)
        elif name == "MapName":
            named_map = self.struct_state[-1]
            assert isinstance(named_map, CiftiNamedMap)
            self.fsm_state.append('MapName')
            self.write_to = 'MapName'
        elif name == "Surface":
            surface = CiftiSurface()
            mim = self.struct_state[-1]
            assert isinstance(mim, CiftiMatrixIndicesMap)
            assert mim.indicesMapToDataType == "CIFTI_INDEX_TYPE_PARCELS"
            surface.brainStructure = attrs["BrainStructure"]
            surface.surfaceNumberOfVertices = int(attrs["SurfaceNumberOfVertices"])
            mim.add_cifti_surface(surface)
        elif name == "Parcel":
            parcel = CiftiParcel()
            mim = self.struct_state[-1]
            assert isinstance(mim, CiftiMatrixIndicesMap)
            parcel.name = attrs["Name"]
            mim.add_cifti_parcel(parcel)
            self.fsm_state.append('Parcel')
            self.struct_state.append(parcel)
        elif name == "Vertices":
            vertices = CiftiVertices()
            parcel = self.struct_state[-1]
            assert isinstance(parcel, CiftiParcel)
            vertices.brainStructure = attrs["BrainStructure"]
            assert vertices.brainStructure in CIFTI_BrainStructures
            parcel.add_cifti_vertices(vertices)
            self.fsm_state.append('Vertices')
            self.struct_state.append(vertices)
            self.write_to = 'Vertices'
        elif name == "VoxelIndicesIJK":
            parent = self.struct_state[-1]
            assert isinstance(parent, (CiftiParcel, CiftiBrainModel))
            parent.voxelIndicesIJK = CiftiVoxelIndicesIJK()

            self.write_to = 'VoxelIndices'
        elif name == "Volume":
            mim = self.struct_state[-1]
            assert isinstance(mim, CiftiMatrixIndicesMap)
            dimensions = tuple([int(val) for val in
                                attrs["VolumeDimensions"].split(',')])
            volume = CiftiVolume(volume_dimensions=dimensions)
            mim.volume = volume
            self.fsm_state.append('Volume')
            self.struct_state.append(volume)
        elif name == "TransformationMatrixVoxelIndicesIJKtoXYZ":
            volume = self.struct_state[-1]
            assert isinstance(volume, CiftiVolume)
            transform = CiftiTransformationMatrixVoxelIndicesIJKtoXYZ()
            transform.meterExponent = int(attrs["MeterExponent"])
            volume.transformationMatrixVoxelIndicesIJKtoXYZ = transform
            self.fsm_state.append('TransformMatrix')
            self.struct_state.append(transform)
            self.write_to = 'TransformMatrix'
        elif name == "BrainModel":
            model = CiftiBrainModel()
            mim = self.struct_state[-1]
            assert isinstance(mim, CiftiMatrixIndicesMap)
            assert mim.indicesMapToDataType == "CIFTI_INDEX_TYPE_BRAIN_MODELS"
            for key, dtype in [("IndexOffset", int),
                               ("IndexCount", int),
                               ("ModelType", str),
                               ("BrainStructure", str),
                               ("SurfaceNumberOfVertices", int)]:
                if key in attrs:
                    var = key[0].lower() + key[1:]
                    setattr(model, var, dtype(attrs[key]))
            assert model.brainStructure in CIFTI_BrainStructures
            assert model.modelType in CIFTI_MODEL_TYPES
            mim.add_cifti_brain_model(model)
            self.fsm_state.append('BrainModel')
            self.struct_state.append(model)
        elif name == "VertexIndices":
            index = CiftiVertexIndices()
            model = self.struct_state[-1]
            assert isinstance(model, CiftiBrainModel)
            self.fsm_state.append('VertexIndices')
            model.vertexIndices = index
            self.struct_state.append(index)
            self.write_to = "VertexIndices"

    def EndElementHandler(self, name):
        self.flush_chardata()
        if DEBUG_PRINT:
            print('End element:\n\t', repr(name))
        if name == 'CIFTI':
            # remove last element of the list
            self.fsm_state.pop()
            self.struct_state.pop()
        elif name == 'Matrix':
            self.fsm_state.pop()
            self.struct_state.pop()
        elif name == 'MetaData':
            self.fsm_state.pop()
            meta = self.struct_state.pop()
            parent = self.struct_state[-1]
            parent.set_metadata(meta)
        elif name == 'MD':
            self.fsm_state.pop()
            pair = self.struct_state.pop()
            meta = self.struct_state[-1]
            if pair.name is not None:
                meta.data.append(pair)
        elif name == 'Name':
            self.write_to = None
        elif name == 'Value':
            self.write_to = None
        elif name == 'MatrixIndicesMap':
            self.fsm_state.pop()
            self.struct_state.pop()
        elif name == 'NamedMap':
            self.fsm_state.pop()
            self.struct_state.pop()
        elif name == 'LabelTable':
            self.fsm_state.pop()
            self.struct_state.pop()
        elif name == 'Label':
            self.fsm_state.pop()
            label = self.struct_state.pop()
            lata = self.struct_state[-1]
            lata.labels.append(label)
            self.write_to = None
        elif name == "MapName":
            self.fsm_state.pop()
            self.write_to = None
        elif name == "Parcel":
            self.fsm_state.pop()
            self.struct_state.pop()
        elif name == "Vertices":
            self.fsm_state.pop()
            self.struct_state.pop()
            self.write_to = None
        elif name == "VoxelIndicesIJK":
            self.write_to = None
        elif name == "Volume":
            self.fsm_state.pop()
            self.struct_state.pop()
        elif name == "TransformationMatrixVoxelIndicesIJKtoXYZ":
            self.fsm_state.pop()
            self.struct_state.pop()
            self.write_to = None
        elif name == "BrainModel":
            self.fsm_state.pop()
            self.struct_state.pop()
        elif name == "VertexIndices":
            self.fsm_state.pop()
            self.struct_state.pop()
            self.write_to = None

    def CharacterDataHandler(self, data):
        """ Collect character data chunks pending collation

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
        """ Collate and process collected character data
        """
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
            data = data.strip()
            pair = self.struct_state[-1]
            assert isinstance(pair, CiftiNVPair)
            pair.name = data
        elif self.write_to == 'Value':
            data = data.strip()
            pair = self.struct_state[-1]
            assert isinstance(pair, CiftiNVPair)
            pair.value = data
        elif self.write_to == 'Vertices':
            # conversion to numpy array
            c = StringIO(data)
            vertices = self.struct_state[-1]
            vertices.vertices = np.genfromtxt(c, dtype=np.int)
            c.close()
        elif self.write_to == 'VoxelIndices':
            # conversion to numpy array
            c = StringIO(data)
            parent = self.struct_state[-1]
            parent.voxelIndicesIJK.indices = np.genfromtxt(c, dtype=np.int)
            c.close()
        elif self.write_to == 'VertexIndices':
            # conversion to numpy array
            c = StringIO(data)
            index = self.struct_state[-1]
            index.indices = np.genfromtxt(c, dtype=np.int)
            c.close()
        elif self.write_to == 'TransformMatrix':
            # conversion to numpy array
            c = StringIO(data)
            transform = self.struct_state[-1]
            transform.matrix = np.genfromtxt(c, dtype=np.float)
            c.close()
        elif self.write_to == 'Label':
            label = self.struct_state[-1]
            label.label = data.strip()
        elif self.write_to == 'MapName':
            named_map = self.struct_state[-1]
            named_map.map_name = data.strip()

    @property
    def pending_data(self):
        " True if there is character data pending for processing "
        return not self._char_blocks is None


def parse_cifti_string(cifti_string, intent_code, filename):
    """ Parse cifti header, return

    Parameters
    ----------
    string : str
        string containing cifti header

    Returns
    -------
    header : cifti header
    """
    if not filename.endswith('nii'):
        raise ValueError('File must be uncompressed to be valid CIFTI')
    parser = ParserCreate()
    HANDLER_NAMES = ['StartElementHandler',
                     'EndElementHandler',
                     'CharacterDataHandler']
    out = Outputter(intent_code=intent_code, filename=filename)
    for name in HANDLER_NAMES:
        setattr(parser, name, getattr(out, name))
    try:
        parser.Parse(cifti_string, True)
    except ExpatError:
        print('An expat error occured while parsing the  Gifti file.')
    return out.header