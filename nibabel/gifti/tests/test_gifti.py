""" Testing gifti objects
"""
import warnings
import sys
from io import BytesIO

import numpy as np

import nibabel as nib
from nibabel.externals.six import string_types
from nibabel.gifti import (GiftiImage, GiftiDataArray, GiftiLabel,
                           GiftiLabelTable, GiftiMetaData, GiftiNVPairs,
                           GiftiCoordSystem)
from nibabel.gifti.gifti import data_tag
from nibabel.nifti1 import data_type_codes
from nibabel.fileholders import FileHolder

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)
from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)
from nibabel.testing import clear_and_catch_warnings
from .test_parse_gifti_fast import (DATA_FILE1, DATA_FILE2, DATA_FILE3,
                                    DATA_FILE4, DATA_FILE5, DATA_FILE6)


def test_gifti_image():
    # Check that we're not modifying the default empty list in the default
    # arguments.
    gi = GiftiImage()
    assert_equal(gi.darrays, [])
    assert_equal(gi.meta.metadata, {})
    assert_equal(gi.labeltable.labels, [])
    arr = np.zeros((2, 3))
    gi.darrays.append(arr)
    # Now check we didn't overwrite the default arg
    gi = GiftiImage()
    assert_equal(gi.darrays, [])

    # Test darrays / numDA
    gi = GiftiImage()
    assert_equal(gi.numDA, 0)

    # Test from numpy numeric array
    data = np.random.random((5,))
    da = GiftiDataArray(data)
    gi.add_gifti_data_array(da)
    assert_equal(gi.numDA, 1)
    assert_array_equal(gi.darrays[0].data, data)

    # Test removing
    gi.remove_gifti_data_array(0)
    assert_equal(gi.numDA, 0)

    # Remove from empty
    gi = GiftiImage()
    gi.remove_gifti_data_array_by_intent(0)
    assert_equal(gi.numDA, 0)

    # Remove one
    gi = GiftiImage()
    da = GiftiDataArray(np.zeros((5,)), intent=0)
    gi.add_gifti_data_array(da)

    gi.remove_gifti_data_array_by_intent(3)
    assert_equal(gi.numDA, 1, "data array should exist on 'missed' remove")

    gi.remove_gifti_data_array_by_intent(da.intent)
    assert_equal(gi.numDA, 0)


def test_gifti_image_bad_inputs():
    img = GiftiImage()
    # Try to set a non-data-array
    assert_raises(TypeError, img.add_gifti_data_array, 'not-a-data-array')

    # Try to set to non-table
    def assign_labeltable(val):
        img.labeltable = val
    assert_raises(TypeError, assign_labeltable, 'not-a-table')

    # Try to set to non-table
    def assign_metadata(val):
        img.meta = val
    assert_raises(TypeError, assign_metadata, 'not-a-meta')


def test_dataarray_empty():
    # Test default initialization of DataArray
    null_da = GiftiDataArray()
    assert_equal(null_da.data, None)
    assert_equal(null_da.intent, 0)
    assert_equal(null_da.datatype, 0)
    assert_equal(null_da.encoding, 3)
    assert_equal(null_da.endian, 2 if sys.byteorder == 'little' else 1)
    assert_equal(null_da.coordsys.dataspace, 0)
    assert_equal(null_da.coordsys.xformspace, 0)
    assert_array_equal(null_da.coordsys.xform, np.eye(4))
    assert_equal(null_da.ind_ord, 1)
    assert_equal(null_da.meta.metadata, {})
    assert_equal(null_da.ext_fname, '')
    assert_equal(null_da.ext_offset, 0)


def test_dataarray_init():
    # Test non-default dataarray initialization
    gda = GiftiDataArray  # shortcut
    assert_equal(gda(None).data, None)
    arr = np.arange(12, dtype=np.float32).reshape((3, 4))
    assert_array_equal(gda(arr).data, arr)
    # Intents
    assert_raises(KeyError, gda, intent=1)  # Invalid code
    assert_raises(KeyError, gda, intent='not an intent')  # Invalid string
    assert_equal(gda(intent=2).intent, 2)
    assert_equal(gda(intent='correlation').intent, 2)
    assert_equal(gda(intent='NIFTI_INTENT_CORREL').intent, 2)
    # Datatype
    assert_equal(gda(datatype=2).datatype, 2)
    assert_equal(gda(datatype='uint8').datatype, 2)
    assert_raises(KeyError, gda, datatype='not_datatype')
    # Float32 datatype comes from array if datatype not set
    assert_equal(gda(arr).datatype, 16)
    # Can be overriden by init
    assert_equal(gda(arr, datatype='uint8').datatype, 2)
    # Encoding
    assert_equal(gda(encoding=1).encoding, 1)
    assert_equal(gda(encoding='ASCII').encoding, 1)
    assert_equal(gda(encoding='GIFTI_ENCODING_ASCII').encoding, 1)
    assert_raises(KeyError, gda, encoding='not an encoding')
    # Endian
    assert_equal(gda(endian=1).endian, 1)
    assert_equal(gda(endian='big').endian, 1)
    assert_equal(gda(endian='GIFTI_ENDIAN_BIG').endian, 1)
    assert_raises(KeyError, gda, endian='not endian code')
    # CoordSys
    aff = np.diag([2, 3, 4, 1])
    cs = GiftiCoordSystem(1, 2, aff)
    da = gda(coordsys=cs)
    assert_equal(da.coordsys.dataspace, 1)
    assert_equal(da.coordsys.xformspace, 2)
    assert_array_equal(da.coordsys.xform, aff)
    # Ordering
    assert_equal(gda(ordering=2).ind_ord, 2)
    assert_equal(gda(ordering='F').ind_ord, 2)
    assert_equal(gda(ordering='ColumnMajorOrder').ind_ord, 2)
    assert_raises(KeyError, gda, ordering='not an ordering')
    # metadata
    meta_dict=dict(one=1, two=2)
    assert_equal(gda(meta=GiftiMetaData.from_dict(meta_dict)).meta.metadata,
                 meta_dict)
    assert_equal(gda(meta=meta_dict).meta.metadata, meta_dict)
    assert_equal(gda(meta=None).meta.metadata, {})
    # ext_fname and ext_offset
    assert_equal(gda(ext_fname='foo').ext_fname, 'foo')
    assert_equal(gda(ext_offset=12).ext_offset, 12)


def test_dataarray_from_array():
    with clear_and_catch_warnings() as w:
        warnings.filterwarnings('always', category=DeprecationWarning)
        da = GiftiDataArray.from_array(np.ones((3, 4)))
        assert_equal(len(w), 1)
        for dt_code in data_type_codes.value_set():
            data_type = data_type_codes.type[dt_code]
            if data_type is np.void:  # not supported
                continue
            arr = np.zeros((10, 3), dtype=data_type)
            da = GiftiDataArray.from_array(arr, 'triangle')
            assert_equal(da.datatype, data_type_codes[arr.dtype])
            bs_arr = arr.byteswap().newbyteorder()
            da = GiftiDataArray.from_array(bs_arr, 'triangle')
            assert_equal(da.datatype, data_type_codes[arr.dtype])


def test_to_xml_open_close_deprecations():
    # Smoke test on deprecated functions
    da = GiftiDataArray(np.ones((1,)), 'triangle')
    with clear_and_catch_warnings() as w:
        warnings.filterwarnings('always', category=DeprecationWarning)
        assert_true(isinstance(da.to_xml_open(), string_types))
        assert_equal(len(w), 1)
    with clear_and_catch_warnings() as w:
        warnings.filterwarnings('once', category=DeprecationWarning)
        assert_true(isinstance(da.to_xml_close(), string_types))
        assert_equal(len(w), 1)


def test_num_dim_deprecation():
    da = GiftiDataArray(np.ones((2, 3, 4)))
    # num_dim is property, set automatically from len(da.dims)
    assert_equal(da.num_dim, 3)
    with clear_and_catch_warnings() as w:
        warnings.filterwarnings('always', category=DeprecationWarning)
        # OK setting num_dim to correct value, but raises DeprecationWarning
        da.num_dim = 3
        assert_equal(len(w), 1)
        # Any other value gives a ValueError
        assert_raises(ValueError, setattr, da, 'num_dim', 4)


def test_labeltable():
    img = GiftiImage()
    assert_equal(len(img.labeltable.labels), 0)

    new_table = GiftiLabelTable()
    new_table.labels += ['test', 'me']
    img.labeltable = new_table
    assert_equal(len(img.labeltable.labels), 2)

    # Test deprecations
    with clear_and_catch_warnings() as w:
        warnings.filterwarnings('always', category=DeprecationWarning)
        newer_table = GiftiLabelTable()
        newer_table.labels += ['test', 'me', 'again']
        img.set_labeltable(newer_table)
        assert_equal(len(w), 1)
        assert_equal(len(img.get_labeltable().labels), 3)
        assert_equal(len(w), 2)


def test_metadata():
    nvpair = GiftiNVPairs('key', 'value')
    md = GiftiMetaData(nvpair=nvpair)
    assert_equal(md.data[0].name, 'key')
    assert_equal(md.data[0].value, 'value')
    # Test deprecation
    with clear_and_catch_warnings() as w:
        warnings.filterwarnings('always', category=DeprecationWarning)
        assert_equal(md.get_metadata(), dict(key='value'))
        assert_equal(len(w), 1)
        assert_equal(len(GiftiDataArray().get_metadata()), 0)
        assert_equal(len(w), 2)


def test_gifti_label_rgba():
    rgba = np.random.rand(4)
    kwargs = dict(zip(['red', 'green', 'blue', 'alpha'], rgba))

    gl1 = GiftiLabel(**kwargs)
    assert_array_equal(rgba, gl1.rgba)

    gl1.red = 2 * gl1.red
    assert_false(np.allclose(rgba, gl1.rgba))  # don't just store the list!

    gl2 = GiftiLabel()
    gl2.rgba = rgba
    assert_array_equal(rgba, gl2.rgba)

    gl2.blue = 2 * gl2.blue
    assert_false(np.allclose(rgba, gl2.rgba))  # don't just store the list!

    def assign_rgba(gl, val):
        gl.rgba = val
    gl3 = GiftiLabel(**kwargs)
    assert_raises(ValueError, assign_rgba, gl3, rgba[:2])
    assert_raises(ValueError, assign_rgba, gl3, rgba.tolist() + rgba.tolist())

    # Test deprecation
    with clear_and_catch_warnings() as w:
        warnings.filterwarnings('once', category=DeprecationWarning)
        assert_equal(kwargs['red'], gl3.get_rgba()[0])
        assert_equal(len(w), 1)

    # Test default value
    gl4 = GiftiLabel()
    assert_equal(len(gl4.rgba), 4)
    assert_true(np.all([elem is None for elem in gl4.rgba]))


def test_print_summary():
    for fil in [DATA_FILE1, DATA_FILE2, DATA_FILE3, DATA_FILE4,
                DATA_FILE5, DATA_FILE6]:
        gimg = nib.load(fil)
        gimg.print_summary()


def test_gifti_coord():
    from ..gifti import GiftiCoordSystem
    gcs = GiftiCoordSystem()
    assert_true(gcs.xform is not None)

    # Smoke test
    gcs.xform = None
    gcs.print_summary()
    gcs.to_xml()


def test_data_tag_deprecated():
    with clear_and_catch_warnings() as w:
        warnings.filterwarnings('once', category=DeprecationWarning)
        data_tag(np.array([]), 'ASCII', '%i', 1)
        assert_equal(len(w), 1)


def test_gifti_round_trip():
    # From section 14.4 in GIFTI Surface Data Format Version 1.0
    # (with some adaptations)

    test_data = b'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE GIFTI SYSTEM "http://www.nitrc.org/frs/download.php/1594/gifti.dtd">
<GIFTI
xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
xsi:noNamespaceSchemaLocation="http://www.nitrc.org/frs/download.php/1303/GIFTI_Caret.xsd"
Version="1.0"
NumberOfDataArrays="2">
<MetaData>
<MD>
<Name><![CDATA[date]]></Name>
<Value><![CDATA[Thu Nov 15 09:05:22 2007]]></Value>
</MD>
</MetaData>
<LabelTable/>
<DataArray Intent="NIFTI_INTENT_POINTSET"
DataType="NIFTI_TYPE_FLOAT32"
ArrayIndexingOrder="RowMajorOrder"
Dimensionality="2"
Dim0="4"
Dim1="3"
Encoding="ASCII"
Endian="LittleEndian"
ExternalFileName=""
ExternalFileOffset="">
<CoordinateSystemTransformMatrix>
<DataSpace><![CDATA[NIFTI_XFORM_TALAIRACH]]></DataSpace>
<TransformedSpace><![CDATA[NIFTI_XFORM_TALAIRACH]]></TransformedSpace>
<MatrixData>
1.000000 0.000000 0.000000 0.000000
0.000000 1.000000 0.000000 0.000000
0.000000 0.000000 1.000000 0.000000
0.000000 0.000000 0.000000 1.000000
</MatrixData>
</CoordinateSystemTransformMatrix>
<Data>
10.5 0 0
0 20.5 0
0 0 30.5
0 0 0
</Data>
</DataArray>
<DataArray Intent="NIFTI_INTENT_TRIANGLE"
DataType="NIFTI_TYPE_INT32"
ArrayIndexingOrder="RowMajorOrder"
Dimensionality="2"
Dim0="4"
Dim1="3"
Encoding="ASCII"
Endian="LittleEndian"
ExternalFileName="" ExternalFileOffset="">
<Data>
0 1 2
1 2 3
0 1 3
0 2 3
</Data>
</DataArray>
</GIFTI>'''

    exp_verts = np.zeros((4, 3))
    exp_verts[0, 0] = 10.5
    exp_verts[1, 1] = 20.5
    exp_verts[2, 2] = 30.5
    exp_faces = np.asarray([[0, 1, 2], [1, 2, 3], [0, 1, 3], [0, 2, 3]],
                           dtype=np.int32)

    def _check_gifti(gio):
        vertices = gio.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
        faces = gio.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
        assert_array_equal(vertices, exp_verts)
        assert_array_equal(faces, exp_faces)

    bio = BytesIO()
    fmap = dict(image=FileHolder(fileobj=bio))

    bio.write(test_data)
    bio.seek(0)
    gio = GiftiImage.from_file_map(fmap)
    _check_gifti(gio)
    # Write and read again
    bio.seek(0)
    gio.to_file_map(fmap)
    bio.seek(0)
    gio2 = GiftiImage.from_file_map(fmap)
    _check_gifti(gio2)


def test_data_array_round_trip():
    # Test valid XML generated from new in-memory array
    # See: https://github.com/nipy/nibabel/issues/469
    verts = np.zeros((4, 3), np.float32)
    verts[0, 0] = 10.5
    verts[1, 1] = 20.5
    verts[2, 2] = 30.5

    vertices = GiftiDataArray(verts)
    img = GiftiImage()
    img.add_gifti_data_array(vertices)
    bio = BytesIO()
    fmap = dict(image=FileHolder(fileobj=bio))
    bio.write(img.to_xml())
    bio.seek(0)
    gio = GiftiImage.from_file_map(fmap)
    vertices = gio.darrays[0].data
    assert_array_equal(vertices, verts)
