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
from ..externals.six import string_types

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
    """ A list of key-value pairs stored in the list self.data """

    def __init__(self, nvpair=None):
        self.data = []
        self.add_metadata(nvpair)

    def _add_remove_metadata(self, metadata, func):
        pairs = []
        if isinstance(metadata, (list, tuple)):
            if isinstance(metadata[0], string_types):
                if len(metadata) != 2:
                    raise ValueError('nvpair must be a 2-list or 2-tuple')
                pairs = [tuple((metadata[0], metadata[1]))]
            else:
                for item in metadata:
                    self._add_remove_metadata(item, func)
                return
        elif isinstance(metadata, dict):
            pairs = metadata.items()
        else:
            raise ValueError('nvpair input must be a list, tuple or dict')
        for pair in pairs:
            if func == 'add':
                if pair not in self.data:
                    self.data.append(pair)
            elif func == 'remove':
                self.data.remove(pair)
            else:
                raise ValueError('Unknown func %s' % func)

    def add_metadata(self, metadata):
        """Add metadata key-value pairs

        This allows storing multiple keys with the same name but different
        values.


        Parameters
        ----------
        metadata : 2-List, 2-Tuple, Dictionary, List[2-List or 2-Tuple]
                     Tuple[2-List or 2-Tuple]

        Returns
        -------
        None

        """
        if metadata is None:
            return
        self._add_remove_metadata(metadata, 'add')

    def remove_metadata(self, metadata):
        if metadata is None:
            return
        self._add_remove_metadata(metadata, 'remove')

    def to_xml(self, prefix='', indent='    '):
        if len(self.data) == 0:
            return ''
        res = "%s<MetaData>\n" % prefix
        preindent = prefix + indent
        for name, value in self.data:
            nvpair = """%s<MD>
%s<Name>%s</Name>
%s<Value>%s</Value>
%s</MD>\n""" % (preindent, preindent + indent, name, preindent + indent,
                value, preindent)
            res += nvpair
        res += "%s</MetaData>\n" % prefix
        return res


class CiftiLabelTable(object):

    def __init__(self):
        self.labels = []

    @property
    def num_labels(self):
        return len(self.labels)

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

    def __init__(self, key = 0, label = '', red = None,\
                  green = None, blue = None, alpha = None):
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
        else:
            raise TypeError("Not a valid CiftiMetaData instance")

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
        else:
            raise TypeError("Not a valid CiftiLabelTable instance")

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
        res = ('%s<Surface BrainStructure="%s" SurfaceNumberOfVertices="%s" '
               '/>\n') % (prefix, self.brainStructure,
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
        self._voxelIndicesIJK = CiftiVoxelIndicesIJK()
        if voxel_indices_ijk is not None:
            self.voxelIndicesIJK = voxel_indices_ijk
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
            raise TypeError("Not a valid CiftiVertices instance")

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
               'IJKtoXYZ MeterExponent="%d">') % (prefix, self.meterExponent)
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
            raise TypeError("Not a valid CiftiBrainModel instance")

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
            raise TypeError("Not a valid CiftiNamedMap instance")

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
            raise TypeError("Not a valid CiftiParcel instance")

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
            raise TypeError("Not a valid CiftiSurface instance")

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
            raise TypeError("Not a valid CiftiVolume instance")

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
        else:
            raise TypeError("Not a valid CiftiMetaData instance")

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
            raise TypeError("Not a valid CiftiMatrixIndicesMap instance")

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

    @classmethod
    def from_header(klass, header=None):
        if header is None:
            return klass()
        if type(header) == klass:
            return header.copy()
        raise ValueError('header is not a CiftiHeader')

class CiftiImage(object):
    header_class = CiftiHeader

    def __init__(self, data=None, header=None, nifti_header=None):
        self.header = CiftiHeader()
        if header is not None:
            self.header = header
        self.data = data
        self.extra = nifti_header

    @classmethod
    def instance_to_filename(klass, img, filename):
        ''' Save `img` in our own format, to name implied by `filename`

        This is a class method

        Parameters
        ----------
        img : ``spatialimage`` instance
           In fact, an object with the API of ``spatialimage`` -
           specifically ``get_data``, ``get_affine``, ``get_header`` and
           ``extra``.
        filename : str
           Filename, implying name to which to save image.
        '''
        img = klass.from_image(img)
        img.to_filename(filename)

    @classmethod
    def from_image(klass, img):
        ''' Class method to create new instance of own class from `img`

        Parameters
        ----------
        img : ``spatialimage`` instance
           In fact, an object with the API of ``spatialimage`` -
           specifically ``get_data``, ``get_affine``, ``get_header`` and
           ``extra``.

        Returns
        -------
        cimg : ``spatialimage`` instance
           Image, of our own class
        '''
        return klass(img._dataobj,
                     klass.header_class.from_header(img.header),
                     extra=img.extra.copy())

    def to_filename(self, filename):
        if not filename.endswith('nii'):
            ValueError('CIFTI files have to be stored as uncompressed NIFTI2')
        from ..nifti2 import Nifti2Image
        from ..nifti1 import Nifti1Extension
        data = np.reshape(self.data, [1, 1, 1, 1] + list(self.data.shape))
        header = self.extra
        extension = Nifti1Extension(32, self.header.to_xml().encode())
        header.extensions.append(extension)
        img = Nifti2Image(data, None, header)
        img.to_filename(filename)


class CiftiDenseDataSeries(CiftiImage):
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
    suffix = '.dtseries.nii'


def load(filename):
    """ Load cifti2 from `filename`

    Parameters
    ----------
    filename : str
        filename of image to be loaded

    Returns
    -------
    img : CiftiImage
        cifti image instance

    Raises
    ------
    ImageFileError: if `filename` doesn't look like cifti
    IOError : if `filename` does not exist
    """
    from ..nifti2 import load as Nifti2load
    return Nifti2load(filename).as_cifti()


def save(img, filename):
    """ Save cifti to `filename`

    Parameters
    ----------
    filename : str
        filename to which to save image
    """
    CiftiImage.instance_to_filename(img, filename)


