# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
from __future__ import division, print_function, absolute_import

import sys
import xml.etree.ElementTree as xml

import numpy as np

from ..nifti1 import data_type_codes, xform_codes, intent_codes
from .util import (array_index_order_codes, gifti_encoding_codes,
                   gifti_endian_codes, KIND2FMT)

# {en,de}codestring in deprecated in Python3, but
# {en,de}codebytes not available in Python2.
# Therefore set the proper functions depending on the Python version.
import base64


class XmlSerializable(object):
    """ Basic interface for serializing an object to xml"""
    def _to_xml_element(self):
        """ Output should be a xml.etree.ElementTree.Element"""
        raise NotImplementedError()

    def to_xml(self, enc='utf-8'):
        """ Output should be an xml string with the given encoding.
        (default: utf-8)"""
        return xml.tostring(self._to_xml_element(), enc)


class GiftiMetaData(XmlSerializable):
    """ A list of GiftiNVPairs in stored in
    the list self.data """
    def __init__(self, nvpair=None):
        self.data = []
        if not nvpair is None:
            self.data.append(nvpair)

    @classmethod
    def from_dict(klass, data_dict):
        meda = klass()
        for k, v in data_dict.items():
            nv = GiftiNVPairs(k, v)
            meda.data.append(nv)
        return meda

    @np.deprecate_with_doc("Use the metadata property instead.")
    def get_metadata(self):
        return self.metadata

    @property
    def metadata(self):
        """ Returns metadata as dictionary """
        self.data_as_dict = {}
        for ele in self.data:
            self.data_as_dict[ele.name] = ele.value
        return self.data_as_dict

    def _to_xml_element(self):
        metadata = xml.Element('MetaData')
        for ele in self.data:
            md = xml.SubElement(metadata, 'MD')
            name = xml.SubElement(md, 'Name')
            value = xml.SubElement(md, 'Value')
            name.text = ele.name
            value.text = ele.value
        return metadata

    def print_summary(self):
        print(self.metadata)


class GiftiNVPairs(object):

    name = str
    value = str

    def __init__(self, name='', value=''):
        self.name = name
        self.value = value


class GiftiLabelTable(XmlSerializable):

    def __init__(self):
        self.labels = []

    def get_labels_as_dict(self):
        self.labels_as_dict = {}
        for ele in self.labels:
            self.labels_as_dict[ele.key] = ele.label
        return self.labels_as_dict

    def _to_xml_element(self):
        labeltable = xml.Element('LabelTable')
        for ele in self.labels:
            label = xml.SubElement(labeltable, 'Label')
            label.attrib['Key'] = str(ele.key)
            label.text = ele.label
            for attr in ['Red', 'Green', 'Blue', 'Alpha']:
                if getattr(ele, attr.lower(), None) is not None:
                    label.attrib[attr] = str(getattr(ele, attr.lower()))
        return labeltable

    def print_summary(self):
        print(self.get_labels_as_dict())


class GiftiLabel(XmlSerializable):
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

    def __init__(self, key=0, label='', red=None, green=None, blue=None,
                 alpha=None):
        self.key = key
        self.label = label
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha

    @np.deprecate_with_doc("Use the rgba property instead.")
    def get_rgba(self):
        return self.rgba

    @property
    def rgba(self):
        """ Returns RGBA as tuple """
        return (self.red, self.green, self.blue, self.alpha)

    @rgba.setter
    def rgba(self, rgba):
        """ Set RGBA via tuple

        Parameters
        ----------
        rgba : tuple (red, green, blue, alpha)

        """
        if len(rgba) != 4:
            raise ValueError('rgba must be length 4.')
        self.red, self.green, self.blue, self.alpha = rgba


def _arr2txt(arr, elem_fmt):
    arr = np.asarray(arr)
    assert arr.dtype.names is None
    if arr.ndim == 1:
        arr = arr[:, None]
    fmt = ' '.join([elem_fmt] * arr.shape[1])
    return '\n'.join(fmt % tuple(row) for row in arr)


class GiftiCoordSystem(XmlSerializable):
    dataspace = int
    xformspace = int
    xform = np.ndarray  # 4x4 numpy array

    def __init__(self, dataspace=0, xformspace=0, xform=None):
        self.dataspace = dataspace
        self.xformspace = xformspace
        if xform is None:
            # create identity matrix
            self.xform = np.identity(4)
        else:
            self.xform = xform

    def _to_xml_element(self):
        coord_xform = xml.Element('CoordinateSystemTransformMatrix')
        if self.xform is not None:
            dataspace = xml.SubElement(coord_xform, 'DataSpace')
            dataspace.text = xform_codes.niistring[self.dataspace]
            xformed_space = xml.SubElement(coord_xform, 'TransformedSpace')
            xformed_space.text = xform_codes.niistring[self.xformspace]
            matrix_data = xml.SubElement(coord_xform, 'MatrixData')
            matrix_data.text = _arr2txt(self.xform, '%10.6f')
        return coord_xform

    def print_summary(self):
        print('Dataspace: ', xform_codes.niistring[self.dataspace])
        print('XFormSpace: ', xform_codes.niistring[self.xformspace])
        print('Affine Transformation Matrix: \n', self.xform)


def _data_tag_element(dataarray, encoding, datatype, ordering):
    """ Creates the data tag depending on the required encoding,
    returns as bytes"""
    import zlib
    ord = array_index_order_codes.npcode[ordering]
    enclabel = gifti_encoding_codes.label[encoding]
    if enclabel == 'ASCII':
        da = _arr2txt(dataarray, datatype)
    elif enclabel in ('B64BIN', 'B64GZ'):
        out = dataarray.tostring(ord)
        if enclabel == 'B64GZ':
            out = zlib.compress(out)
        da = base64.b64encode(out).decode()
    elif enclabel == 'External':
        raise NotImplementedError("In what format are the external files?")
    else:
        da = ''

    data = xml.Element('Data')
    data.text = da
    return data


def data_tag(dataarray, encoding, datatype, ordering):
    data = _data_tag_element(dataarray, encoding, datatype, ordering)
    return xml.tostring(data, 'utf-8')


class GiftiDataArray(XmlSerializable):

    # These are for documentation only; we don't use these class variables
    intent = int
    datatype = int
    ind_ord = int
    num_dim = int
    dims = list
    encoding = int
    endian = int
    ext_fname = str
    ext_offset = str
    data = np.ndarray
    coordsys = GiftiCoordSystem
    meta = GiftiMetaData

    def __init__(self, data=None):
        self.data = data
        self.dims = []
        self.meta = GiftiMetaData()
        self.coordsys = GiftiCoordSystem()
        self.ext_fname = ''
        self.ext_offset = ''

    @classmethod
    def from_array(klass,
                   darray,
                   intent,
                   datatype=None,
                   encoding="GIFTI_ENCODING_B64GZ",
                   endian=sys.byteorder,
                   coordsys=None,
                   ordering="C",
                   meta=None):
        """ Creates a new Gifti data array

        Parameters
        ----------
        darray : ndarray
            NumPy data array
        intent : string
            NIFTI intent code, see nifti1.intent_codes
        datatype : None or string, optional
            NIFTI data type codes, see nifti1.data_type_codes
            If None, the datatype of the NumPy array is taken.
        encoding : string, optionaal
            Encoding of the data, see util.gifti_encoding_codes;
            default: GIFTI_ENCODING_B64GZ
        endian : string, optional
            The Endianness to store the data array.  Should correspond to the
            machine endianness.  default: system byteorder
        coordsys : GiftiCoordSystem, optional
            If None, a identity transformation is taken.
        ordering : string, optional
            The ordering of the array. see util.array_index_order_codes;
            default: RowMajorOrder - C ordering
        meta : None or dict, optional
            A dictionary for metadata information.  If None, gives empty dict.

        Returns
        -------
        da : instance of our own class
        """
        if meta is None:
            meta = {}
        cda = klass(darray)
        cda.num_dim = len(darray.shape)
        cda.dims = list(darray.shape)
        if datatype is None:
            cda.datatype = data_type_codes.code[darray.dtype]
        else:
            cda.datatype = data_type_codes.code[datatype]
        cda.intent = intent_codes.code[intent]
        cda.encoding = gifti_encoding_codes.code[encoding]
        cda.endian = gifti_endian_codes.code[endian]
        if not coordsys is None:
            cda.coordsys = coordsys
        cda.ind_ord = array_index_order_codes.code[ordering]
        cda.meta = GiftiMetaData.from_dict(meta)
        return cda

    def _to_xml_element(self):
        # fix endianness to machine endianness
        self.endian = gifti_endian_codes.code[sys.byteorder]

        data_array = xml.Element('DataArray', attrib={
            'Intent': intent_codes.niistring[self.intent],
            'DataType': data_type_codes.niistring[self.datatype],
            'ArrayIndexingOrder': array_index_order_codes.label[self.ind_ord],
            'Dimensionality': str(self.num_dim),
            'Encoding': gifti_encoding_codes.specs[self.encoding],
            'Endian': gifti_endian_codes.specs[self.endian],
            'ExternalFileName': self.ext_fname,
            'ExternalFileOffset': self.ext_offset})
        for di, dn in enumerate(self.dims):
            data_array.attrib['Dim%d' % di] = str(dn)

        if self.meta is not None:
            data_array.append(self.meta._to_xml_element())
        if self.coordsys is not None:
            data_array.append(self.coordsys._to_xml_element())
        # write data array depending on the encoding
        dt_kind = data_type_codes.dtype[self.datatype].kind
        data_array.append(
            _data_tag_element(self.data,
                              gifti_encoding_codes.specs[self.encoding],
                              KIND2FMT[dt_kind],
                              self.ind_ord))

        return data_array

    def print_summary(self):
        print('Intent: ', intent_codes.niistring[self.intent])
        print('DataType: ', data_type_codes.niistring[self.datatype])
        print('ArrayIndexingOrder: ',
              array_index_order_codes.label[self.ind_ord])
        print('Dimensionality: ', self.num_dim)
        print('Dimensions: ', self.dims)
        print('Encoding: ', gifti_encoding_codes.specs[self.encoding])
        print('Endian: ', gifti_endian_codes.specs[self.endian])
        print('ExternalFileName: ', self.ext_fname)
        print('ExternalFileOffset: ', self.ext_offset)
        if not self.coordsys is None:
            print('----')
            print('Coordinate System:')
            print(self.coordsys.print_summary())

    @np.deprecate_with_doc("Use the metadata property instead.")
    def get_metadata(self):
        return self.meta.metadata

    @property
    def metadata(self):
        """ Returns metadata as dictionary """
        return self.meta.metadata


class GiftiImage(XmlSerializable):

    def __init__(self, meta=None, labeltable=None, darrays=None,
                 version="1.0"):
        if darrays is None:
            darrays = []
        if meta is None:
            meta = GiftiMetaData()
        if labeltable is None:
            labeltable = GiftiLabelTable()

        self._labeltable = labeltable
        self._meta = meta

        self.darrays = darrays
        self.version = version

    @property
    def numDA(self):
        return len(self.darrays)

    @property
    def labeltable(self):
        return self._labeltable

    @labeltable.setter
    def labeltable(self, labeltable):
        """ Set the labeltable for this GiftiImage

        Parameters
        ----------
        labeltable : GiftiLabelTable

        """
        if not isinstance(labeltable, GiftiLabelTable):
            raise TypeError("Not a valid GiftiLabelTable instance")
        self._labeltable = labeltable

    @np.deprecate_with_doc("Use the gifti_img.labeltable property instead.")
    def set_labeltable(self, labeltable):
        self.labeltable = labeltable

    @np.deprecate_with_doc("Use the gifti_img.labeltable property instead.")
    def get_labeltable(self):
        return self.labeltable

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, meta):
        """ Set the metadata for this GiftiImage

        Parameters
        ----------
        meta : GiftiMetaData

        Returns
        -------
        None
        """
        if not isinstance(meta, GiftiMetaData):
            raise TypeError("Not a valid GiftiMetaData instance")
        self._meta = meta

    @np.deprecate_with_doc("Use the gifti_img.labeltable property instead.")
    def set_metadata(self, meta):
        self.meta = meta

    @np.deprecate_with_doc("Use the gifti_img.labeltable property instead.")
    def get_meta(self):
        return self.meta

    def add_gifti_data_array(self, dataarr):
        """ Adds a data array to the GiftiImage

        Parameters
        ----------
        dataarr : GiftiDataArray
        """
        if not isinstance(dataarr, GiftiDataArray):
            raise TypeError("Not a valid GiftiDataArray instance")
        self.darrays.append(dataarr)

    def remove_gifti_data_array(self, ith):
        """ Removes the ith data array element from the GiftiImage """
        self.darrays.pop(ith)

    def remove_gifti_data_array_by_intent(self, intent):
        """ Removes all the data arrays with the given intent type """
        intent2remove = intent_codes.code[intent]
        for dele in self.darrays:
            if dele.intent == intent2remove:
                self.darrays.remove(dele)

    def get_arrays_from_intent(self, intent):
        """ Returns a a list of GiftiDataArray elements matching
        the given intent """

        it = intent_codes.code[intent]

        return [x for x in self.darrays if x.intent == it]

    @np.deprecate_with_doc("Use get_arrays_from_intent instead.")
    def getArraysFromIntent(self, intent):
        return self.get_arrays_from_intent(intent)

    def print_summary(self):
        print('----start----')
        print('Source filename: ', self.filename)
        print('Number of data arrays: ', self.numDA)
        print('Version: ', self.version)
        if self.meta is not None:
            print('----')
            print('Metadata:')
            print(self.meta.print_summary())
        if self.labeltable is not None:
            print('----')
            print('Labeltable:')
            print(self.labeltable.print_summary())
        for i, da in enumerate(self.darrays):
            print('----')
            print('DataArray %s:' % i)
            print(da.print_summary())
        print('----end----')


    def _to_xml_element(self):
        GIFTI = xml.Element('GIFTI', attrib={
            'Version': self.version,
            'NumberOfDataArrays': str(self.numDA)})
        if self.meta is not None:
            GIFTI.append(self.meta._to_xml_element())
        if self.labeltable is not None:
            GIFTI.append(self.labeltable._to_xml_element())
        for dar in self.darrays:
            GIFTI.append(dar._to_xml_element())
        return GIFTI

    def to_xml(self, enc='utf-8'):
        """ Return XML corresponding to image content """
        return b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE GIFTI SYSTEM "http://www.nitrc.org/frs/download.php/115/gifti.dtd">
""" + XmlSerializable.to_xml(self, enc)
