''' Testing trackvis module '''
from __future__ import division, print_function, absolute_import

from functools import partial

import numpy as np

from ..externals.six import BytesIO
from .. import trackvis as tv
from ..orientations import aff2axcodes
from ..volumeutils import native_code, swapped_code

from nose.tools import assert_true, assert_false, assert_equal, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal
from ..testing import error_warnings, suppress_warnings


def test_write():
    streams = []
    out_f = BytesIO()
    tv.write(out_f, [], {})
    assert_equal(out_f.getvalue(), tv.empty_header().tostring())
    out_f.truncate(0)
    out_f.seek(0)
    # Write something not-default
    tv.write(out_f, [], {'id_string': 'TRACKb'})
    # read it back
    out_f.seek(0)
    streams, hdr = tv.read(out_f)
    assert_equal(hdr['id_string'], b'TRACKb')
    # check that we can pass none for the header
    out_f.truncate(0)
    out_f.seek(0)
    tv.write(out_f, [])
    out_f.truncate(0)
    out_f.seek(0)
    tv.write(out_f, [], None)
    # check that we check input values
    out_f.truncate(0)
    out_f.seek(0)
    assert_raises(tv.HeaderError,
                  tv.write, out_f, [], {'id_string': 'not OK'})
    assert_raises(tv.HeaderError,
                  tv.write, out_f, [], {'version': 3})
    assert_raises(tv.HeaderError,
                  tv.write, out_f, [], {'hdr_size': 0})


def test_write_scalars_props():
    # Test writing of scalar array with streamlines
    N = 6
    M = 2
    P = 4
    points = np.arange(N * 3).reshape((N, 3))
    scalars = np.arange(N * M).reshape((N, M)) + 100
    props = np.arange(P) + 1000
    # If scalars not same size for each point, error
    out_f = BytesIO()
    streams = [(points, None, None),
               (points, scalars, None)]
    assert_raises(tv.DataError, tv.write, out_f, streams)
    out_f.seek(0)
    streams = [(points, np.zeros((N, M + 1)), None),
               (points, scalars, None)]
    assert_raises(tv.DataError, tv.write, out_f, streams)
    # Or if scalars different N compared to points
    bad_scalars = np.zeros((N + 1, M))
    out_f.seek(0)
    streams = [(points, bad_scalars, None),
               (points, bad_scalars, None)]
    assert_raises(tv.DataError, tv.write, out_f, streams)
    # Similarly properties must have the same length for each streamline
    out_f.seek(0)
    streams = [(points, scalars, None),
               (points, scalars, props)]
    assert_raises(tv.DataError, tv.write, out_f, streams)
    out_f.seek(0)
    streams = [(points, scalars, np.zeros((P + 1,))),
               (points, scalars, props)]
    assert_raises(tv.DataError, tv.write, out_f, streams)
    # If all is OK, then we get back what we put in
    out_f.seek(0)
    streams = [(points, scalars, props),
               (points, scalars, props)]
    tv.write(out_f, streams)
    out_f.seek(0)
    back_streams, hdr = tv.read(out_f)
    for actual, expected in zip(streams, back_streams):
        for a_el, e_el in zip(actual, expected):
            assert_array_equal(a_el, e_el)
    # Also so if the datatype of points, scalars is already float32 (github
    # issue #53)
    out_f.seek(0)
    streams = [(points.astype('f4'),
                scalars.astype('f4'),
                props.astype('f4'))]
    tv.write(out_f, streams)
    out_f.seek(0)
    back_streams, hdr = tv.read(out_f)
    for actual, expected in zip(streams, back_streams):
        for a_el, e_el in zip(actual, expected):
            assert_array_almost_equal(a_el, e_el)


def streams_equal(stream1, stream2):
    if not np.all(stream1[0] == stream2[0]):
        return False
    if stream1[1] is None:
        if not stream2[1] is None:
            return False
    if stream1[2] is None:
        if not stream2[2] is None:
            return False
    if not np.all(stream1[1] == stream2[1]):
        return False
    if not np.all(stream1[2] == stream2[2]):
        return False
    return True


def streamlist_equal(streamlist1, streamlist2):
    if len(streamlist1) != len(streamlist2):
        return False
    for s1, s2 in zip(streamlist1, streamlist2):
        if not streams_equal(s1, s2):
            return False
    return True


def test_round_trip():
    out_f = BytesIO()
    xyz0 = np.tile(np.arange(5).reshape(5, 1), (1, 3))
    xyz1 = np.tile(np.arange(5).reshape(5, 1) + 10, (1, 3))
    streams = [(xyz0, None, None), (xyz1, None, None)]
    tv.write(out_f, streams, {})
    out_f.seek(0)
    streams2, hdr = tv.read(out_f)
    assert_true(streamlist_equal(streams, streams2))
    # test that we can write in different endianness and get back same result,
    # for versions 1, 2 and not-specified
    for in_dict, back_version in (({}, 2),
                                  ({'version': 2}, 2),
                                  ({'version': 1}, 1)):
        for endian_code in (native_code, swapped_code):
            out_f.seek(0)
            tv.write(out_f, streams, in_dict, endian_code)
            out_f.seek(0)
            streams2, hdr = tv.read(out_f)
            assert_true(streamlist_equal(streams, streams2))
            assert_equal(hdr['version'], back_version)
    # test that we can get out and pass in generators
    out_f.seek(0)
    streams3, hdr = tv.read(out_f, as_generator=True)
    # check this is a generator rather than a list
    assert_true(hasattr(streams3, 'send'))
    # but that it results in the same output
    assert_true(streamlist_equal(streams, list(streams3)))
    # write back in
    out_f.seek(0)
    streams3, hdr = tv.read(out_f, as_generator=True)
    # Now we need a new file object, because we're still using the old one for
    # our generator
    out_f_write = BytesIO()
    tv.write(out_f_write, streams3, {})
    # and re-read just to check
    out_f_write.seek(0)
    streams2, hdr = tv.read(out_f_write)
    assert_true(streamlist_equal(streams, streams2))


def test_points_processing():
    # We may need to process points if they are in voxel or mm format
    out_f = BytesIO()

    def _rt(streams, hdr, points_space):
        # run round trip through IO object
        out_f.seek(0)
        tv.write(out_f, streams, hdr, points_space=points_space)
        out_f.seek(0)
        res0 = tv.read(out_f)
        out_f.seek(0)
        return res0, tv.read(out_f, points_space=points_space)
    n_pts = 5
    ijk0 = np.arange(n_pts * 3).reshape((n_pts, 3)) / 2.0
    ijk1 = ijk0 + 20
    # Check with and without some scalars
    for scalars in ((None, None),
                    (np.arange(n_pts)[:, None],
                     np.arange(n_pts)[:, None] + 99)):
        vx_streams = [(ijk0, scalars[0], None), (ijk1, scalars[1], None)]
        vxmm_streams = [(ijk0 * [[2, 3, 4]], scalars[0], None),
                        (ijk1 * [[2, 3, 4]], scalars[1], None)]
        # voxmm is the default.  In this case we don't do anything to the
        # points, and we let the header pass through without further checks
        (raw_streams, hdr), (proc_streams, _) = _rt(vxmm_streams, {}, None)
        assert_true(streamlist_equal(raw_streams, proc_streams))
        assert_true(streamlist_equal(vxmm_streams, proc_streams))
        (raw_streams, hdr), (proc_streams, _) = _rt(vxmm_streams, {}, 'voxmm')
        assert_true(streamlist_equal(raw_streams, proc_streams))
        assert_true(streamlist_equal(vxmm_streams, proc_streams))
        # with 'voxels' as input, check for not all voxel_size == 0, warn if any
        # voxel_size == 0
        for hdr in (  # these cause read / write errors
            # empty header has 0 voxel sizes
            {},
            {'voxel_size': [0, 0, 0]},  # the default
            {'voxel_size': [-2, 3, 4]},  # negative not valid
        ):
            # Check error on write
            out_f.seek(0)
            assert_raises(tv.HeaderError,
                          tv.write, out_f, vx_streams, hdr, None, 'voxel')
            out_f.seek(0)
            # bypass write error and check read
            tv.write(out_f, vxmm_streams, hdr, None, points_space=None)
            out_f.seek(0)
            assert_raises(tv.HeaderError, tv.read, out_f, False, 'voxel')
        # There's a warning for any voxel sizes == 0
        hdr = {'voxel_size': [2, 3, 0]}
        with error_warnings():
            assert_raises(UserWarning, _rt, vx_streams, hdr, 'voxel')
        # This should be OK
        hdr = {'voxel_size': [2, 3, 4]}
        (raw_streams, hdr), (proc_streams, _) = _rt(vx_streams, hdr, 'voxel')
        assert_true(streamlist_equal(vxmm_streams, raw_streams))
        assert_true(streamlist_equal(vx_streams, proc_streams))
        # Now we try with rasmm points.  In this case we need valid voxel_size,
        # and voxel_order, and vox_to_ras.  The voxel_order has to match the
        # vox_to_ras, and so do the voxel sizes
        aff = np.diag([2, 3, 4, 1])
        # In this case the trk -> vx and vx -> mm invert each other
        rasmm_streams = vxmm_streams
        for hdr in (  # all these cause read and write errors for rasmm
            # Empty header has no valid affine
            {},
            # Error if ras_to_mm not defined (as in version 1)
            {'voxel_size': [2, 3, 4], 'voxel_order': 'RAS', 'version': 1},
            # or it's all zero
            {'voxel_size': [2, 3, 4], 'voxel_order': 'RAS',
             'vox_to_ras': np.zeros((4, 4))},
            # as it is by default
            {'voxel_size': [2, 3, 4], 'voxel_order': 'RAS'},
            # or the voxel_size doesn't match the affine
            {'voxel_size': [2, 2, 4], 'voxel_order': 'RAS',
             'vox_to_ras': aff},
            # or the voxel_order doesn't match the affine
            {'voxel_size': [2, 3, 4], 'voxel_order': 'LAS',
             'vox_to_ras': aff},
        ):
            # Check error on write
            out_f.seek(0)
            assert_raises(tv.HeaderError,
                          tv.write, out_f, rasmm_streams, hdr, None, 'rasmm')
            out_f.seek(0)
            # bypass write error and check read
            tv.write(out_f, vxmm_streams, hdr, None, points_space=None)
            out_f.seek(0)
            assert_raises(tv.HeaderError, tv.read, out_f, False, 'rasmm')
        # This should be OK
        hdr = {'voxel_size': [2, 3, 4], 'voxel_order': 'RAS',
               'vox_to_ras': aff}
        (raw_streams, hdr), (proc_streams, _) = _rt(rasmm_streams, hdr, 'rasmm')
        assert_true(streamlist_equal(vxmm_streams, raw_streams))
        assert_true(streamlist_equal(rasmm_streams, proc_streams))
        # More complex test to check matrix orientation
        fancy_affine = np.array([[0., -2, 0, 10],
                                 [3, 0, 0, 20],
                                 [0, 0, 4, 30],
                                 [0, 0, 0, 1]])
        hdr = {'voxel_size': [3, 2, 4], 'voxel_order': 'ALS',
               'vox_to_ras': fancy_affine}

        def f(pts):  # from vx to mm
            pts = pts[:, [1, 0, 2]] * [[-2, 3, 4]]  # apply zooms / reorder
            return pts + [[10, 20, 30]]  # apply translations
        xyz0, xyz1 = f(ijk0), f(ijk1)
        fancy_rasmm_streams = [(xyz0, scalars[0], None),
                               (xyz1, scalars[1], None)]
        fancy_vxmm_streams = [(ijk0 * [[3, 2, 4]], scalars[0], None),
                              (ijk1 * [[3, 2, 4]], scalars[1], None)]
        (raw_streams, hdr), (proc_streams, _) = _rt(
            fancy_rasmm_streams, hdr, 'rasmm')
        assert_true(streamlist_equal(fancy_vxmm_streams, raw_streams))
        assert_true(streamlist_equal(fancy_rasmm_streams, proc_streams))


def test__check_hdr_points_space():
    # Test checking routine for points_space input given header
    # None or voxmm -> no checks, pass through
    assert_equal(tv._check_hdr_points_space({}, None), None)
    assert_equal(tv._check_hdr_points_space({}, 'voxmm'), None)
    # strange value for points_space -> ValueError
    assert_raises(ValueError,
                  tv._check_hdr_points_space, {}, 'crazy')
    # Input not in (None, 'voxmm', 'voxels', 'rasmm') - error
    # voxels means check voxel sizes present and not all 0.
    hdr = tv.empty_header()
    assert_array_equal(hdr['voxel_size'], [0, 0, 0])
    assert_raises(tv.HeaderError,
                  tv._check_hdr_points_space, hdr, 'voxel')
    # Negative voxel size gives error - because it is not what trackvis does,
    # and this not what we mean by 'voxmm'
    hdr['voxel_size'] = [-2, 3, 4]
    assert_raises(tv.HeaderError,
                  tv._check_hdr_points_space, hdr, 'voxel')
    # Warning here only
    hdr['voxel_size'] = [2, 3, 0]
    with error_warnings():
        assert_raises(UserWarning,
                      tv._check_hdr_points_space, hdr, 'voxel')
    # This is OK
    hdr['voxel_size'] = [2, 3, 4]
    assert_equal(tv._check_hdr_points_space(hdr, 'voxel'), None)
    # rasmm - check there is an affine, that it matches voxel_size and
    # voxel_order
    # no affine
    hdr['voxel_size'] = [2, 3, 4]
    assert_raises(tv.HeaderError,
                  tv._check_hdr_points_space, hdr, 'rasmm')
    # still no affine
    hdr['voxel_order'] = 'RAS'
    assert_raises(tv.HeaderError,
                  tv._check_hdr_points_space, hdr, 'rasmm')
    # nearly an affine, but 0 at position 3,3 - means not recorded in trackvis
    # standard
    hdr['vox_to_ras'] = np.diag([2, 3, 4, 0])
    assert_raises(tv.HeaderError,
                  tv._check_hdr_points_space, hdr, 'rasmm')
    # This affine doesn't match RAS voxel order
    hdr['vox_to_ras'] = np.diag([-2, 3, 4, 1])
    assert_raises(tv.HeaderError,
                  tv._check_hdr_points_space, hdr, 'rasmm')
    # This affine doesn't match the voxel size
    hdr['vox_to_ras'] = np.diag([3, 3, 4, 1])
    assert_raises(tv.HeaderError,
                  tv._check_hdr_points_space, hdr, 'rasmm')
    # This should be OK
    good_aff = np.diag([2, 3, 4, 1])
    hdr['vox_to_ras'] = good_aff
    assert_equal(tv._check_hdr_points_space(hdr, 'rasmm'),
                 None)
    # Default voxel order of LPS assumed
    hdr['voxel_order'] = ''
    # now the RAS affine raises an error
    assert_raises(tv.HeaderError,
                  tv._check_hdr_points_space, hdr, 'rasmm')
    # this affine does have LPS voxel order
    good_lps = np.dot(np.diag([-1, -1, 1, 1]), good_aff)
    hdr['vox_to_ras'] = good_lps
    assert_equal(tv._check_hdr_points_space(hdr, 'rasmm'),
                 None)


def test_empty_header():
    for endian in '<>':
        for version in (1, 2):
            hdr = tv.empty_header(endian, version)
            assert_equal(hdr['id_string'], b'TRACK')
            assert_equal(hdr['version'], version)
            assert_equal(hdr['hdr_size'], 1000)
            assert_array_equal(
                hdr['image_orientation_patient'],
                [0, 0, 0, 0, 0, 0])
    hdr = tv.empty_header(version=2)
    assert_array_equal(hdr['vox_to_ras'], np.zeros((4, 4)))
    hdr_endian = tv.endian_codes[tv.empty_header().dtype.byteorder]
    assert_equal(hdr_endian, tv.native_code)


def test_get_affine():
    # Test get affine behavior, including pending deprecation
    hdr = tv.empty_header()
    # Using version 1 affine is not a good idea because is fragile and not
    # very useful. The default atleast_v2=None mode raises a FutureWarning
    with error_warnings():
        assert_raises(FutureWarning, tv.aff_from_hdr, hdr)
    # testing the old behavior
    old_afh = partial(tv.aff_from_hdr, atleast_v2=False)
    # default header gives useless affine
    assert_array_equal(old_afh(hdr),
                       np.diag([0, 0, 0, 1]))
    hdr['voxel_size'] = 1
    assert_array_equal(old_afh(hdr),
                       np.diag([0, 0, 0, 1]))
    # DICOM direction cosines
    hdr['image_orientation_patient'] = [1, 0, 0, 0, 1, 0]
    assert_array_equal(old_afh(hdr),
                       np.diag([-1, -1, 1, 1]))
    # RAS direction cosines
    hdr['image_orientation_patient'] = [-1, 0, 0, 0, -1, 0]
    assert_array_equal(old_afh(hdr),
                       np.eye(4))
    # translations
    hdr['origin'] = [1, 2, 3]
    exp_aff = np.eye(4)
    exp_aff[:3, 3] = [-1, -2, 3]
    assert_array_equal(old_afh(hdr),
                       exp_aff)
    # check against voxel order.  This one works
    hdr['voxel_order'] = ''.join(aff2axcodes(exp_aff))
    assert_equal(hdr['voxel_order'], b'RAS')
    assert_array_equal(old_afh(hdr), exp_aff)
    # This one doesn't
    hdr['voxel_order'] = 'LAS'
    assert_raises(tv.HeaderError, old_afh, hdr)
    # This one does work because the routine allows the final dimension to
    # be flipped to try and match the voxel order
    hdr['voxel_order'] = 'RAI'
    exp_aff = exp_aff * [[1, 1, -1, 1]]
    assert_array_equal(old_afh(hdr), exp_aff)
    # Check round trip case for flipped and unflipped, when positive voxels
    # only allowed.  This checks that the flipping heuristic works.
    flipped_aff = exp_aff
    unflipped_aff = exp_aff * [1, 1, -1, 1]
    for in_aff, o_codes in ((unflipped_aff, b'RAS'),
                            (flipped_aff, b'RAI')):
        hdr = tv.empty_header()
        tv.aff_to_hdr(in_aff, hdr, pos_vox=True, set_order=True)
        # Unset easier option
        hdr['vox_to_ras'] = 0
        assert_equal(hdr['voxel_order'], o_codes)
        # Check it came back the way we wanted
        assert_array_equal(old_afh(hdr), in_aff)
    # Check that the default case matches atleast_v2=False case
    with suppress_warnings():
        assert_array_equal(tv.aff_from_hdr(hdr), flipped_aff)
    # now use the easier vox_to_ras field
    hdr = tv.empty_header()
    aff = np.eye(4)
    aff[:3, :] = np.arange(12).reshape(3, 4)
    hdr['vox_to_ras'] = aff
    # Pass v2 flag explicitly to avoid warnings
    assert_array_equal(tv.aff_from_hdr(hdr, atleast_v2=False), aff)
    # mappings work too
    d = {'version': 1,
         'voxel_size': np.array([1, 2, 3]),
         'image_orientation_patient': np.array([1, 0, 0, 0, 1, 0]),
         'origin': np.array([10, 11, 12])}
    aff = tv.aff_from_hdr(d, atleast_v2=False)


def test_aff_to_hdr():
    # The behavior is changing soon, change signaled by FutureWarnings
    # This is the call to get the old behavior
    old_a2h = partial(tv.aff_to_hdr, pos_vox=False, set_order=False)
    hdr = {'version': 1}
    affine = np.diag([1, 2, 3, 1])
    affine[:3, 3] = [10, 11, 12]
    old_a2h(affine, hdr)
    assert_array_almost_equal(tv.aff_from_hdr(hdr, atleast_v2=False), affine)
    # put flip into affine
    aff2 = affine.copy()
    aff2[:, 2] *= -1
    old_a2h(aff2, hdr)
    # Historically we flip the first axis if there is a negative determinant
    assert_array_almost_equal(hdr['voxel_size'], [-1, 2, 3])
    assert_array_almost_equal(tv.aff_from_hdr(hdr, atleast_v2=False), aff2)
    # Test that default mode raises DeprecationWarning
    with error_warnings():
        assert_raises(FutureWarning, tv.aff_to_hdr, affine, hdr)
        assert_raises(FutureWarning, tv.aff_to_hdr, affine, hdr, None, None)
        assert_raises(FutureWarning, tv.aff_to_hdr, affine, hdr, False, None)
        assert_raises(FutureWarning, tv.aff_to_hdr, affine, hdr, None, False)
    # And has same effect as above
    with suppress_warnings():
        tv.aff_to_hdr(affine, hdr)
    assert_array_almost_equal(tv.aff_from_hdr(hdr, atleast_v2=False), affine)
    # Check pos_vox and order flags
    for hdr in ({}, {'version': 2}, {'version': 1}):
        tv.aff_to_hdr(aff2, hdr, pos_vox=True, set_order=False)
        assert_array_equal(hdr['voxel_size'], [1, 2, 3])
        assert_false('voxel_order' in hdr)
        tv.aff_to_hdr(aff2, hdr, pos_vox=False, set_order=True)
        assert_array_equal(hdr['voxel_size'], [-1, 2, 3])
        assert_equal(hdr['voxel_order'], 'RAI')
        tv.aff_to_hdr(aff2, hdr, pos_vox=True, set_order=True)
        assert_array_equal(hdr['voxel_size'], [1, 2, 3])
        assert_equal(hdr['voxel_order'], 'RAI')
        if 'version' in hdr and hdr['version'] == 1:
            assert_false('vox_to_ras' in hdr)
        else:
            assert_array_equal(hdr['vox_to_ras'], aff2)


def test_tv_class():
    tvf = tv.TrackvisFile([])
    assert_equal(tvf.streamlines, [])
    assert_true(isinstance(tvf.header, np.ndarray))
    assert_equal(tvf.endianness, tv.native_code)
    assert_equal(tvf.filename, None)
    out_f = BytesIO()
    tvf.to_file(out_f)
    assert_equal(out_f.getvalue(), tv.empty_header().tostring())
    out_f.truncate(0)
    out_f.seek(0)
    # Write something not-default
    tvf = tv.TrackvisFile([], {'id_string': 'TRACKb'})
    tvf.to_file(out_f)
    # read it back
    out_f.seek(0)
    tvf_back = tv.TrackvisFile.from_file(out_f)
    assert_equal(tvf_back.header['id_string'], b'TRACKb')
    # check that we check input values
    out_f.truncate(0)
    out_f.seek(0)
    assert_raises(tv.HeaderError,
                  tv.TrackvisFile,
                  [], {'id_string': 'not OK'})
    assert_raises(tv.HeaderError,
                  tv.TrackvisFile,
                  [], {'version': 3})
    assert_raises(tv.HeaderError,
                  tv.TrackvisFile,
                  [], {'hdr_size': 0})
    affine = np.diag([1, 2, 3, 1])
    affine[:3, 3] = [10, 11, 12]
    # affine methods will raise same warnings and errors as function
    with error_warnings():
        assert_raises(FutureWarning, tvf.set_affine, affine)
        assert_raises(FutureWarning, tvf.set_affine, affine, None, None)
        assert_raises(FutureWarning, tvf.set_affine, affine, False, None)
        assert_raises(FutureWarning, tvf.set_affine, affine, None, False)
        assert_raises(FutureWarning, tvf.get_affine)
        assert_raises(FutureWarning, tvf.get_affine, None)
    tvf.set_affine(affine, pos_vox=True, set_order=True)
    aff = tvf.get_affine(atleast_v2=True)
    assert_array_almost_equal(aff, affine)
    # Test that we raise an error with an iterator
    assert_raises(tv.TrackvisFileError,
                  tv.TrackvisFile,
                  iter([]))


def test_tvfile_io():
    # Test reading and writing tracks with file class
    out_f = BytesIO()
    ijk0 = np.arange(15).reshape((5, 3)) / 2.0
    ijk1 = ijk0 + 20
    vx_streams = [(ijk0, None, None), (ijk1, None, None)]
    vxmm_streams = [(ijk0 * [[2, 3, 4]], None, None),
                    (ijk1 * [[2, 3, 4]], None, None)]
    # Roundtrip basic
    tvf = tv.TrackvisFile(vxmm_streams)
    tvf.to_file(out_f)
    out_f.seek(0)
    tvf2 = tv.TrackvisFile.from_file(out_f)
    assert_equal(tvf2.filename, None)
    assert_true(streamlist_equal(vxmm_streams, tvf2.streamlines))
    assert_equal(tvf2.points_space, None)
    # Voxel points_space
    tvf = tv.TrackvisFile(vx_streams, points_space='voxel')
    out_f.seek(0)
    # No voxel size - error
    assert_raises(tv.HeaderError, tvf.to_file, out_f)
    out_f.seek(0)
    # With voxel size, no error, roundtrip works
    tvf.header['voxel_size'] = [2, 3, 4]
    tvf.to_file(out_f)
    out_f.seek(0)
    tvf2 = tv.TrackvisFile.from_file(out_f, points_space='voxel')
    assert_true(streamlist_equal(vx_streams, tvf2.streamlines))
    assert_equal(tvf2.points_space, 'voxel')
    out_f.seek(0)
    # Also with affine specified
    tvf = tv.TrackvisFile(vx_streams, points_space='voxel',
                          affine=np.diag([2, 3, 4, 1]))
    tvf.to_file(out_f)
    out_f.seek(0)
    tvf2 = tv.TrackvisFile.from_file(out_f, points_space='voxel')
    assert_true(streamlist_equal(vx_streams, tvf2.streamlines))
    # Fancy affine test
    fancy_affine = np.array([[0., -2, 0, 10],
                             [3, 0, 0, 20],
                             [0, 0, 4, 30],
                             [0, 0, 0, 1]])

    def f(pts):  # from vx to mm
        pts = pts[:, [1, 0, 2]] * [[-2, 3, 4]]  # apply zooms / reorder
        return pts + [[10, 20, 30]]  # apply translations
    xyz0, xyz1 = f(ijk0), f(ijk1)
    fancy_rasmm_streams = [(xyz0, None, None), (xyz1, None, None)]
    # Roundtrip
    tvf = tv.TrackvisFile(fancy_rasmm_streams, points_space='rasmm')
    out_f.seek(0)
    # No affine
    assert_raises(tv.HeaderError, tvf.to_file, out_f)
    out_f.seek(0)
    # With affine set, no error, roundtrip works
    tvf.set_affine(fancy_affine, pos_vox=True, set_order=True)
    tvf.to_file(out_f)
    out_f.seek(0)
    tvf2 = tv.TrackvisFile.from_file(out_f, points_space='rasmm')
    assert_true(streamlist_equal(fancy_rasmm_streams, tvf2.streamlines))
    assert_equal(tvf2.points_space, 'rasmm')
    out_f.seek(0)
    # Also when affine given in init
    tvf = tv.TrackvisFile(fancy_rasmm_streams, points_space='rasmm',
                          affine=fancy_affine)
    tvf.to_file(out_f)
    out_f.seek(0)
    tvf2 = tv.TrackvisFile.from_file(out_f, points_space='rasmm')
    assert_true(streamlist_equal(fancy_rasmm_streams, tvf2.streamlines))


def test_read_truncated():
    # Test behavior when last track contains fewer points than specified
    out_f = BytesIO()
    xyz0 = np.tile(np.arange(5).reshape(5, 1), (1, 3))
    xyz1 = np.tile(np.arange(5).reshape(5, 1) + 10, (1, 3))
    streams = [(xyz0, None, None), (xyz1, None, None)]
    tv.write(out_f, streams, {})
    # Truncate the last stream by one point
    value = out_f.getvalue()[:-(3 * 4)]
    new_f = BytesIO(value)
    # By default, raises a DataError
    assert_raises(tv.DataError, tv.read, new_f)
    # This corresponds to strict mode
    new_f.seek(0)
    assert_raises(tv.DataError, tv.read, new_f, strict=True)
    # lenient error mode lets this error pass, with truncated track
    short_streams = [(xyz0, None, None), (xyz1[:-1], None, None)]
    new_f.seek(0)
    streams2, hdr = tv.read(new_f, strict=False)
    assert_true(streamlist_equal(streams2, short_streams))
    # Check that lenient works when number of tracks is 0, where 0 signals to
    # the reader to read until the end of the file.
    again_hdr = hdr.copy()
    assert_equal(again_hdr['n_count'], 2)
    again_hdr['n_count'] = 0
    again_bytes = again_hdr.tostring() + value[again_hdr.itemsize:]
    again_f = BytesIO(again_bytes)
    streams2, _ = tv.read(again_f, strict=False)
    assert_true(streamlist_equal(streams2, short_streams))
    # Set count to one above actual number of tracks, always raise error
    again_hdr['n_count'] = 3
    again_bytes = again_hdr.tostring() + value[again_hdr.itemsize:]
    again_f = BytesIO(again_bytes)
    assert_raises(tv.DataError, tv.read, again_f, strict=False)
