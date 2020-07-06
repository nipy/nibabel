# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Classes defining Gifti objects

The Gifti specification was (at time of writing) available as a PDF download
from http://www.nitrc.org/projects/gifti/
"""

import sys
import numpy as np
import base64

from .. import xmlutils as xml
from ..filebasedimages import SerializableImage
from ..nifti1 import data_type_codes, xform_codes, intent_codes
from .util import (array_index_order_codes, gifti_encoding_codes,
                   gifti_endian_codes, KIND2FMT)
from ..deprecated import deprecate_with_version


class GiftiMetaData(xml.XmlSerializable):
    """ A sequence of GiftiNVPairs containing metadata for a gifti data array
    """

    def __init__(self, nvpair=None):
        self.data = []
        if nvpair is not None:
            self.data.append(nvpair)

    @classmethod
    def from_dict(klass, data_dict):
        meda = klass()
        for k, v in data_dict.items():
            nv = GiftiNVPairs(k, v)
            meda.data.append(nv)
        return meda

    @deprecate_with_version(
        'get_metadata method deprecated. '
        "Use the metadata property instead."
        '2.1', '4.0')
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
    """ Gifti name / value pairs

    Attributes
    ----------
    name : str
    value : str
    """
    def __init__(self, name=u'', value=u''):
        self.name = name
        self.value = value


class GiftiLabelTable(xml.XmlSerializable):
    """ Gifti label table: a sequence of key, label pairs

    From the gifti spec dated 2011-01-14:
        The label table is used by DataArrays whose values are an key into the
        LabelTable's labels. A file should contain at most one LabelTable and
        it must be located in the file prior to any DataArray elements.
    """

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


class GiftiLabel(xml.XmlSerializable):
    """ Gifti label: association of integer key with optional RGBA values

    Quotes are from the gifti spec dated 2011-01-14.

    Attributes
    ----------
    key : int
        (From the spec): "This required attribute contains a non-negative
        integer value. If a DataArray's Intent is NIFTI_INTENT_LABEL and a
        value in the DataArray is 'X', its corresponding label is the label
        with the Key attribute containing the value 'X'. In early versions of
        the GIFTI file format, the attribute Index was used instead of Key. If
        an Index attribute is encountered, it should be processed like the Key
        attribute."
    red : None or float
        Optional value for red.
    green : None or float
        Optional value for green.
    blue : None or float
        Optional value for blue.
    alpha : None or float
        Optional value for alpha.

    Notes
    -----
    freesurfer examples seem not to conform to datatype "NIFTI_TYPE_RGBA32"
    because they are floats, not 4 8-bit integers.
    """

    def __init__(self, key=0, red=None, green=None, blue=None, alpha=None):
        self.key = key
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha

    @deprecate_with_version(
        'get_rgba method deprecated. '
        "Use the rgba property instead."
        '2.1', '4.0')
    def get_rgba(self):
        return self.rgba

    @property
    def rgba(self):
        """ Returns RGBA as tuple """
        return (self.red, self.green, self.blue, self.alpha)

    @rgba.setter
    def rgba(self, rgba):
        """ Set RGBA via sequence

        Parameters
        ----------
        rgba : length 4 sequence
            Sequence containing values for red, green, blue, alpha.
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


class GiftiCoordSystem(xml.XmlSerializable):
    """ Gifti coordinate system transform matrix

    Quotes are from the gifti spec dated 2011-01-14.

        "For a DataArray with an Intent NIFTI_INTENT_POINTSET, this element
        describes the stereotaxic space of the data before and after the
        application of a transformation matrix. The most common stereotaxic
        space is the Talairach Space that places the origin at the anterior
        commissure and the negative X, Y, and Z axes correspond to left,
        posterior, and inferior respectively.  At least one
        CoordinateSystemTransformMatrix is required in a DataArray with an
        intent of NIFTI_INTENT_POINTSET. Multiple
        CoordinateSystemTransformMatrix elements may be used to describe the
        transformation to multiple spaces."

    Attributes
    ----------
    dataspace : int
        From the spec: Contains the stereotaxic space of a DataArray's data
        prior to application of the transformation matrix. The stereotaxic
        space should be one of:

          - NIFTI_XFORM_UNKNOWN
          - NIFTI_XFORM_SCANNER_ANAT
          - NIFTI_XFORM_ALIGNED_ANAT
          - NIFTI_XFORM_TALAIRACH
          - NIFTI_XFORM_MNI_152

    xformspace : int
        Spec: "Contains the stereotaxic space of a DataArray's data after
        application of the transformation matrix. See the DataSpace element for
        a list of stereotaxic spaces."

    xform : array-like shape (4, 4)
        Affine transformation matrix
    """

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


@deprecate_with_version(
    "data_tag is an internal API that will be discontinued.",
    '2.1', '4.0')
def data_tag(dataarray, encoding, datatype, ordering):
    class DataTag(xml.XmlSerializable):

        def __init__(self, *args):
            self.args = args

        def _to_xml_element(self):
            return _data_tag_element(*self.args)

    return DataTag(dataarray, encoding, datatype, ordering).to_xml()


def _data_tag_element(dataarray, encoding, dtype, ordering):
    """ Creates data tag with given `encoding`, returns as XML element
    """
    import zlib
    order = array_index_order_codes.npcode[ordering]
    enclabel = gifti_encoding_codes.label[encoding]
    if enclabel == 'ASCII':
        # XXX Accommodating data_tag API
        # On removal (nibabel 4.0) drop str case
        da = _arr2txt(dataarray, dtype if isinstance(dtype, str) else KIND2FMT[dtype.kind])
    elif enclabel in ('B64BIN', 'B64GZ'):
        # XXX Accommodating data_tag API - don't try to fix dtype
        if isinstance(dtype, str):
            dtype = dataarray.dtype
        out = np.asanyarray(dataarray, dtype).tobytes(order)
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


class GiftiDataArray(xml.XmlSerializable):
    """ Container for Gifti numerical data array and associated metadata

    Quotes are from the gifti spec dated 2011-01-14.

    Description of DataArray in spec:
        "This element contains the numeric data and its related metadata. The
        CoordinateSystemTransformMatrix child is only used when the DataArray's
        Intent is NIFTI_INTENT_POINTSET.  FileName and FileOffset are required
        if the data is stored in an external file."

    Attributes
    ----------
    darray : None or ndarray
        Data array
    intent : int
        NIFTI intent code, see nifti1.intent_codes
    datatype : int
        NIFTI data type codes, see nifti1.data_type_codes.  From the spec:
        "This required attribute describes the numeric type of the data
        contained in a Data Array and are limited to the types displayed in the
        table:

        NIFTI_TYPE_UINT8 : Unsigned, 8-bit bytes.
        NIFTI_TYPE_INT32 : Signed, 32-bit integers.
        NIFTI_TYPE_FLOAT32 : 32-bit single precision floating point."

        At the moment, we do not enforce that the datatype is one of these
        three.
    encoding : string
        Encoding of the data, see util.gifti_encoding_codes; default is
        GIFTI_ENCODING_B64GZ.
    endian : string
        The Endianness to store the data array.  Should correspond to the
        machine endianness.  Default is system byteorder.
    coordsys : :class:`GiftiCoordSystem` instance
        Input and output coordinate system with tranformation matrix between
        the two.
    ind_ord : int
        The ordering of the array. see util.array_index_order_codes.  Default
        is RowMajorOrder - C ordering
    meta : :class:`GiftiMetaData` instance
        An instance equivalent to a dictionary for metadata information.
    ext_fname : str
        Filename in which data is stored, or empty string if no corresponding
        filename.
    ext_offset : int
        Position in bytes within `ext_fname` at which to start reading data.
    """

    def __init__(self,
                 data=None,
                 intent='NIFTI_INTENT_NONE',
                 datatype=None,
                 encoding="GIFTI_ENCODING_B64GZ",
                 endian=sys.byteorder,
                 coordsys=None,
                 ordering="C",
                 meta=None,
                 ext_fname=u'',
                 ext_offset=0):
        """
        Returns a shell object that cannot be saved.
        """
        self.data = None if data is None else np.asarray(data)
        self.intent = intent_codes.code[intent]
        if datatype is None:
            datatype = 'none' if self.data is None else self.data.dtype
        self.datatype = data_type_codes.code[datatype]
        self.encoding = gifti_encoding_codes.code[encoding]
        self.endian = gifti_endian_codes.code[endian]
        self.coordsys = coordsys or GiftiCoordSystem()
        self.ind_ord = array_index_order_codes.code[ordering]
        self.meta = (GiftiMetaData() if meta is None else
                     meta if isinstance(meta, GiftiMetaData) else
                     GiftiMetaData.from_dict(meta))
        self.ext_fname = ext_fname
        self.ext_offset = ext_offset
        self.dims = [] if self.data is None else list(self.data.shape)

    @property
    def num_dim(self):
        return len(self.dims)

    # Setter for backwards compatibility with pymvpa
    @num_dim.setter
    @deprecate_with_version(
        "num_dim will be read-only in future versions of nibabel",
        '2.1', '4.0')
    def num_dim(self, value):
        if value != len(self.dims):
            raise ValueError(f'num_dim value {value} != number of '
                             f'dimensions len(self.dims) {len(self.dims)}')

    @classmethod
    @deprecate_with_version(
        'from_array method is deprecated. '
        'Please use GiftiDataArray constructor instead.',
        '2.1', '4.0')
    def from_array(klass,
                   darray,
                   intent="NIFTI_INTENT_NONE",
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
        return klass(data=darray,
                     intent=intent,
                     datatype=datatype,
                     encoding=encoding,
                     endian=endian,
                     coordsys=coordsys,
                     ordering=ordering,
                     meta=meta)

    def _to_xml_element(self):
        # fix endianness to machine endianness
        self.endian = gifti_endian_codes.code[sys.byteorder]

        # All attribute values must be strings
        data_array = xml.Element('DataArray', attrib={
            'Intent': intent_codes.niistring[self.intent],
            'DataType': data_type_codes.niistring[self.datatype],
            'ArrayIndexingOrder': array_index_order_codes.label[self.ind_ord],
            'Dimensionality': str(self.num_dim),
            'Encoding': gifti_encoding_codes.specs[self.encoding],
            'Endian': gifti_endian_codes.specs[self.endian],
            'ExternalFileName': self.ext_fname,
            'ExternalFileOffset': str(self.ext_offset)})
        for di, dn in enumerate(self.dims):
            data_array.attrib['Dim%d' % di] = str(dn)

        if self.meta is not None:
            data_array.append(self.meta._to_xml_element())
        if self.coordsys is not None:
            data_array.append(self.coordsys._to_xml_element())
        # write data array depending on the encoding
        data_array.append(
            _data_tag_element(self.data,
                              gifti_encoding_codes.specs[self.encoding],
                              data_type_codes.dtype[self.datatype],
                              self.ind_ord))

        return data_array

    @deprecate_with_version(
        'to_xml_open method deprecated. '
        'Use the to_xml() function instead.',
        '2.1', '4.0')
    def to_xml_open(self):
        out = """<DataArray Intent="%s"
\tDataType="%s"
\tArrayIndexingOrder="%s"
\tDimensionality="%s"
%s\tEncoding="%s"
\tEndian="%s"
\tExternalFileName="%s"
\tExternalFileOffset="%d">\n"""
        di = ""
        for i, n in enumerate(self.dims):
            di = di + f'\tDim{i}="{n}\"\n'
        return out % (intent_codes.niistring[self.intent],
                      data_type_codes.niistring[self.datatype],
                      array_index_order_codes.label[self.ind_ord],
                      str(self.num_dim),
                      str(di),
                      gifti_encoding_codes.specs[self.encoding],
                      gifti_endian_codes.specs[self.endian],
                      self.ext_fname,
                      self.ext_offset,
                      )

    @deprecate_with_version(
        'to_xml_close method deprecated. '
        'Use the to_xml() function instead.',
        '2.1', '4.0')
    def to_xml_close(self):
        return "</DataArray>\n"

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
        if self.coordsys is not None:
            print('----')
            print('Coordinate System:')
            print(self.coordsys.print_summary())

    @deprecate_with_version(
        'get_metadata method deprecated. '
        "Use the metadata property instead."
        '2.1', '4.0')
    def get_metadata(self):
        return self.meta.metadata

    @property
    def metadata(self):
        """ Returns metadata as dictionary """
        return self.meta.metadata


class GiftiImage(xml.XmlSerializable, SerializableImage):
    """ GIFTI image object

    The Gifti spec suggests using the following suffixes to your
    filename when saving each specific type of data:

    .gii
        Generic GIFTI File
    .coord.gii
        Coordinates
    .func.gii
        Functional
    .label.gii
        Labels
    .rgba.gii
        RGB or RGBA
    .shape.gii
        Shape
    .surf.gii
        Surface
    .tensor.gii
        Tensors
    .time.gii
        Time Series
    .topo.gii
        Topology

    The Gifti file is stored in endian convention of the current machine.
    """
    valid_exts = ('.gii',)
    files_types = (('image', '.gii'),)

    # The parser will in due course be a GiftiImageParser, but we can't set
    # that now, because it would result in a circular import.  We set it after
    # the class has been defined, at the end of the class definition.
    parser = None

    def __init__(self, header=None, extra=None, file_map=None, meta=None,
                 labeltable=None, darrays=None, version=u"1.0"):
        super(GiftiImage, self).__init__(header=header, extra=extra,
                                         file_map=file_map)
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
        labeltable : :class:`GiftiLabelTable` instance
        """
        if not isinstance(labeltable, GiftiLabelTable):
            raise TypeError("Not a valid GiftiLabelTable instance")
        self._labeltable = labeltable

    @deprecate_with_version(
        'set_labeltable method deprecated. '
        "Use the gifti_img.labeltable property instead.",
        '2.1', '4.0')
    def set_labeltable(self, labeltable):
        self.labeltable = labeltable

    @deprecate_with_version(
        'get_labeltable method deprecated. '
        "Use the gifti_img.labeltable property instead.",
        '2.1', '4.0')
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
        meta : :class:`GiftiMetaData` instance
        """
        if not isinstance(meta, GiftiMetaData):
            raise TypeError("Not a valid GiftiMetaData instance")
        self._meta = meta

    @deprecate_with_version(
        'set_meta method deprecated. '
        "Use the gifti_img.meta property instead.",
        '2.1', '4.0')
    def set_metadata(self, meta):
        self.meta = meta

    @deprecate_with_version(
        'get_meta method deprecated. '
        "Use the gifti_img.meta property instead.",
        '2.1', '4.0')
    def get_meta(self):
        return self.meta

    def add_gifti_data_array(self, dataarr):
        """ Adds a data array to the GiftiImage

        Parameters
        ----------
        dataarr : :class:`GiftiDataArray` instance
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
        """ Return list of GiftiDataArray elements matching given intent
        """
        it = intent_codes.code[intent]
        return [x for x in self.darrays if x.intent == it]

    def agg_data(self, intent_code=None):
        """
        Aggregate GIFTI data arrays into an ndarray or tuple of ndarray

        In the general case, the numpy data array is extracted from each ``GiftiDataArray``
        object and returned in a ``tuple``, in the order they are found in the GIFTI image.

        If all ``GiftiDataArray`` s have ``intent`` of 2001 (``NIFTI_INTENT_TIME_SERIES``),
        then the data arrays are concatenated as columns, producing a vertex-by-time array.
        If an ``intent_code`` is passed, data arrays are filtered by the selected intents,
        before being aggregated.
        This may be useful for images containing several intents, or ensuring an expected
        data type in an image of uncertain provenance.
        If ``intent_code`` is a ``tuple``, then a ``tuple`` will be returned with the result of
        ``agg_data`` for each element, in order.
        This may be useful for ensuring that expected data arrives in a consistent order.

        Parameters
        ----------
        intent_code : None, string, integer or tuple of strings or integers, optional
            code(s) specifying nifti intent

        Returns
        -------
        tuple of ndarrays or ndarray
            If the input is a tuple, the returned tuple will match the order.

        Examples
        --------

        Consider a surface GIFTI file:

        >>> import nibabel as nib
        >>> from nibabel.testing import test_data
        >>> surf_img = nib.load(test_data('gifti', 'ascii.gii'))

        The coordinate data, which is indicated by the ``NIFTI_INTENT_POINTSET``
        intent code, may be retrieved using any of the following equivalent
        calls:

        >>> coords = surf_img.agg_data('NIFTI_INTENT_POINTSET')
        >>> coords_2 = surf_img.agg_data('pointset')
        >>> coords_3 = surf_img.agg_data(1008)  # Numeric code for pointset
        >>> print(np.array2string(coords, precision=3))
        [[-16.072 -66.188  21.267]
         [-16.706 -66.054  21.233]
         [-17.614 -65.402  21.071]]
        >>> np.array_equal(coords, coords_2)
        True
        >>> np.array_equal(coords, coords_3)
        True

        Similarly, the triangle mesh can be retrieved using various intent
        specifiers:

        >>> triangles = surf_img.agg_data('NIFTI_INTENT_TRIANGLE')
        >>> triangles_2 = surf_img.agg_data('triangle')
        >>> triangles_3 = surf_img.agg_data(1009)  # Numeric code for pointset
        >>> print(np.array2string(triangles))
        [0 1 2]
        >>> np.array_equal(triangles, triangles_2)
        True
        >>> np.array_equal(triangles, triangles_3)
        True

        All arrays can be retrieved as a ``tuple`` by omitting the intent
        code:

        >>> coords_4, triangles_4 = surf_img.agg_data()
        >>> np.array_equal(coords, coords_4)
        True
        >>> np.array_equal(triangles, triangles_4)
        True

        Finally, a tuple of intent codes may be passed in order to select
        the arrays in a specific order:

        >>> triangles_5, coords_5 = surf_img.agg_data(('triangle', 'pointset'))
        >>> np.array_equal(triangles, triangles_5)
        True
        >>> np.array_equal(coords, coords_5)
        True

        The following image is a GIFTI file with ten (10) data arrays of the same
        size, and with intent code 2001 (``NIFTI_INTENT_TIME_SERIES``):

        >>> func_img = nib.load(test_data('gifti', 'task.func.gii'))

        When aggregating time series data, these arrays are concatenated into
        a single, vertex-by-timestep array:

        >>> series = func_img.agg_data()
        >>> series.shape
        (642, 10)

        In the case of a GIFTI file with unknown data arrays, it may be preferable
        to specify the intent code, so that a time series array is always returned:

        >>> series_2 = func_img.agg_data('NIFTI_INTENT_TIME_SERIES')
        >>> series_3 = func_img.agg_data('time series')
        >>> series_4 = func_img.agg_data(2001)
        >>> np.array_equal(series, series_2)
        True
        >>> np.array_equal(series, series_3)
        True
        >>> np.array_equal(series, series_4)
        True

        Requesting a data array from a GIFTI file with no matching intent codes
        will result in an empty tuple:

        >>> surf_img.agg_data('time series')
        ()
        >>> func_img.agg_data('triangle')
        ()
        """

        # Allow multiple intents to specify the order
        # e.g., agg_data(('pointset', 'triangle')) ensures consistent order

        if isinstance(intent_code, tuple):
            return tuple(self.agg_data(intent_code=code) for code in intent_code)

        darrays = self.darrays if intent_code is None else self.get_arrays_from_intent(intent_code)
        all_data = tuple(da.data for da in darrays)
        all_intent = {intent_codes.niistring[da.intent] for da in darrays}

        if all_intent == {'NIFTI_INTENT_TIME_SERIES'}:  # stack when the gifti is a timeseries
            return np.column_stack(all_data)

        if len(all_data) == 1:
            all_data = all_data[0]

        return all_data

    @deprecate_with_version(
        'getArraysFromIntent method deprecated. '
        "Use get_arrays_from_intent instead.",
        '2.1', '4.0')
    def getArraysFromIntent(self, intent):
        return self.get_arrays_from_intent(intent)

    def print_summary(self):
        print('----start----')
        print('Source filename: ', self.get_filename())
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
            print(f'DataArray {i}:')
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
""" + xml.XmlSerializable.to_xml(self, enc)

    # Avoid the indirection of going through to_file_map
    to_bytes = to_xml

    def to_file_map(self, file_map=None):
        """ Save the current image to the specified file_map

        Parameters
        ----------
        file_map : dict
            Dictionary with single key ``image`` with associated value which is
            a :class:`FileHolder` instance pointing to the image file.

        Returns
        -------
        None
        """
        if file_map is None:
            file_map = self.file_map
        f = file_map['image'].get_prepare_fileobj('wb')
        f.write(self.to_xml())

    @classmethod
    def from_file_map(klass, file_map, buffer_size=35000000):
        """ Load a Gifti image from a file_map

        Parameters
        ----------
        file_map : dict
            Dictionary with single key ``image`` with associated value which is
            a :class:`FileHolder` instance pointing to the image file.

        Returns
        -------
        img : GiftiImage
        """
        parser = klass.parser(buffer_size=buffer_size)
        parser.parse(fptr=file_map['image'].get_prepare_fileobj('rb'))
        return parser.img

    @classmethod
    def from_filename(klass, filename, buffer_size=35000000):
        file_map = klass.filespec_to_file_map(filename)
        img = klass.from_file_map(file_map, buffer_size=buffer_size)
        return img


# Now GiftiImage is defined, we can import the parser module and set the parser
from .parse_gifti_fast import GiftiImageParser
GiftiImage.parser = GiftiImageParser
