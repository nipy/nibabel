# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
from __future__ import division, print_function, absolute_import

from distutils.version import LooseVersion

from ..externals.six import StringIO
from xml.parsers.expat import ParserCreate, ExpatError

import numpy as np

from .cifti import (CiftiMetaData, CiftiImage, CiftiHeader, CiftiLabel,
                    CiftiLabelTable, CiftiVertexIndices,
                    CiftiVoxelIndicesIJK, CiftiBrainModel, CiftiMatrix,
                    CiftiMatrixIndicesMap, CiftiNamedMap, CiftiParcel,
                    CiftiSurface, CiftiTransformationMatrixVoxelIndicesIJKtoXYZ,
                    CiftiVertices, CiftiVolume, CIFTI_BrainStructures,
                    CIFTI_MODEL_TYPES,
                    CiftiDenseDataSeries)

DEBUG_PRINT = False

class Outputter(object):

    def __init__(self):
        # finite state machine stack
        self.fsm_state = []
        self.struct_state = []

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
            self.header.version = attrs['Version']
            if LooseVersion(self.header.version) < LooseVersion('2'):
                raise ValueError('Only CIFTI-2 files are supported')
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
            pair = ['', '']
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
            if pair[0]:
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
            pair[0] = data
        elif self.write_to == 'Value':
            data = data.strip()
            pair = self.struct_state[-1]
            pair[1] = data
        elif self.write_to == 'Vertices':
            # conversion to numpy array
            c = StringIO(data.strip().decode('utf-8'))
            vertices = self.struct_state[-1]
            vertices.vertices = np.genfromtxt(c, dtype=np.int)
            c.close()
        elif self.write_to == 'VoxelIndices':
            # conversion to numpy array
            c = StringIO(data.strip().decode('utf-8'))
            parent = self.struct_state[-1]
            parent.voxelIndicesIJK.indices = np.genfromtxt(c, dtype=np.int)
            c.close()
        elif self.write_to == 'VertexIndices':
            # conversion to numpy array
            c = StringIO(data.strip())
            index = self.struct_state[-1]
            index.indices = np.genfromtxt(c, dtype=np.int)
            c.close()
        elif self.write_to == 'TransformMatrix':
            # conversion to numpy array
            c = StringIO(data.strip())
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


def parse_cifti_string(cifti_string):
    """ Parse cifti header, return

    Parameters
    ----------
    string : str
        string containing cifti header

    Returns
    -------
    header : cifti header
    """
    parser = ParserCreate()
    HANDLER_NAMES = ['StartElementHandler',
                     'EndElementHandler',
                     'CharacterDataHandler']
    out = Outputter()
    for name in HANDLER_NAMES:
        setattr(parser, name, getattr(out, name))
    parser.Parse(cifti_string, True)
    return out.header

def create_cifti_image(nifti2_image, cifti_header, intent_code):
    data = np.squeeze(nifti2_image.get_data())
    nifti_header = nifti2_image.get_header()
    for ext in nifti_header.extensions:
        if ext.get_code() == 32:
            nifti_header.extensions.remove(ext)
            break
    header = parse_cifti_string(cifti_header)
    if intent_code == 3002:
        img = CiftiDenseDataSeries(data, header, nifti_header)
    else:
        img = CiftiImage(data, header, nifti_header)
    return img
