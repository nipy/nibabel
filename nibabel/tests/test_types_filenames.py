''' Tests for filename container '''

from StringIO import StringIO

from nibabel.filename_parser import types_filenames, TypesFilenamesError

from nose.tools import assert_equal, assert_true, assert_false, \
     assert_raises

from nibabel.testing import parametric


@parametric
def test_filenames():
    types_exts = (('image', '.img'), ('header', '.hdr'))
    for t_fname in ('test.img', 'test.hdr', 'test', 'test.'):
        tfns = types_filenames(t_fname, types_exts)
        yield assert_equal(tfns,
                           {'image': 'test.img',
                            'header': 'test.hdr'})
    # enforcing extensions raises an error for bad extension
    yield assert_raises(TypesFilenamesError,
                        types_filenames,
                        'test.funny',
                        types_exts)
    # If not enforcing extensions, it does the best job it can,
    # assuming the passed filename is for the first type (in this case
    # 'image') 
    tfns = types_filenames('test.funny', types_exts,
                           enforce_extensions=False)
    yield assert_equal(tfns,
                       {'header': 'test.hdr',
                        'image': 'test.funny'})
    # .gz and .bz2 suffixes to extensions, by default, are removed
    # before extension checking etc, and then put back onto every
    # returned filename. 
    tfns = types_filenames('test.img.gz', types_exts)
    yield assert_equal(tfns,
                       {'header': 'test.hdr.gz',
                        'image': 'test.img.gz'})
    tfns = types_filenames('test.img.bz2', types_exts)
    yield assert_equal(tfns,
                       {'header': 'test.hdr.bz2',
                        'image': 'test.img.bz2'})
    # of course, if we don't know about e.g. gz, and enforce_extensions
    # is on, we get an errror
    yield assert_raises(TypesFilenamesError,
                        types_filenames,
                        'test.img.gz',
                        types_exts, ())
    # if we don't know about .gz extension, and not enforcing, then we
    # get something a bit odd
    tfns = types_filenames('test.img.gz', types_exts,
                           trailing_suffixes=(),
                           enforce_extensions=False)
    yield assert_equal(tfns,
                       {'header': 'test.img.hdr',
                        'image': 'test.img.gz'})
    # the suffixes we remove and replaces can be any suffixes. 
    tfns = types_filenames('test.img.bzr', types_exts, ('.bzr',))
    yield assert_equal(tfns,
                       {'header': 'test.hdr.bzr',
                        'image': 'test.img.bzr'})
    # If we specifically pass the remove / replace suffixes, then we
    # don't remove / replace the .gz and .bz2, unless they are passed
    # specifically.
    tfns = types_filenames('test.img.bzr', types_exts,
                           trailing_suffixes=('.bzr',),
                           enforce_extensions=False)
    yield assert_equal(tfns,
                       {'header': 'test.hdr.bzr',
                        'image': 'test.img.bzr'})
    # but, just .gz or .bz2 as extension gives an error, if enforcing is on
    yield assert_raises(TypesFilenamesError,
                        types_filenames,
                        'test.gz',
                        types_exts)
    yield assert_raises(TypesFilenamesError,
                        types_filenames,
                        'test.bz2',
                        types_exts)
    # if enforcing is off, it tries to work out what the other files
    # should be assuming the passed filename is of the first input type
    tfns = types_filenames('test.gz', types_exts,
                           enforce_extensions=False)
    yield assert_equal(tfns,
                       {'image': 'test.gz',
                        'header': 'test.hdr.gz'})
