''' Tests for filename container '''

from StringIO import StringIO

from nose.tools import assert_equals, assert_true, assert_false, \
     assert_raises

from nibabel.filetuples import FileTuples, FileTuplesError


def test_init():
    fn = FileTuples()
    yield assert_equals, fn.types, ()
    types = (('image','.img'),('header','.hdr'))
    fn = FileTuples(types=types)
    yield assert_equals, fn.types, types
    yield assert_equals, fn.default_type, 'image'
    fn = FileTuples(types=types,
                   default_type='header')
    yield assert_equals, fn.default_type, 'header'
    yield assert_equals, fn.ignored_suffixes, ()
    fn = FileTuples(types=types,
                   default_type='image',
                   ignored_suffixes=('.bz2', '.gz'))
    yield assert_equals, fn.ignored_suffixes, ('.bz2', '.gz')
    yield assert_equals, fn.enforce_extensions, True    
    fn = FileTuples(types=types,
                   default_type='image',
                   ignored_suffixes=('.bz2', '.gz'),
                   enforce_extensions=False)
    yield assert_equals, fn.enforce_extensions, False


def test_filenames():
    fn = FileTuples()
    fn.ignored_suffixes=()
    fn.add_type('image', '.img')
    fn.add_type('header', '.hdr')
    fn.default_type = 'image'
    fn.enforce_extensions = True
    for t_fname in ('test.img', 'test.hdr', 'test', 'test.'):
        fn.set_filenames(t_fname)
        yield assert_equals, fn.get_filenames(), ('test.img', 'test.hdr')
        yield assert_equals, fn.get_file_of_type('header'), 'test.hdr'
        yield assert_equals, fn.get_file_of_type('image'), 'test.img'
        # check if all the files are set
        yield assert_equals, fn.get_file_of_type('all'), ('test.img', 'test.hdr')
    yield assert_raises, FileTuplesError, fn.set_filenames, 'test.funny'
    # If not enforcing extensions, it does the best job it can,
    # assuming the passed filename is for the image
    fn.enforce_extensions = False
    fn.set_filenames('test.funny')
    yield assert_equals, fn.get_file_of_type('header'), 'test.hdr'
    yield assert_equals, fn.get_file_of_type('image'), 'test.funny'
    # .gz and .bz2 suffixes to extensions can be made OK
    # but give odd behavior by default
    fn.enforce_extensions = True
    yield assert_raises, FileTuplesError, fn.set_filenames, 'test.img.gz'
    fn.enforce_extensions = False
    fn.set_filenames('test.img.gz')
    yield assert_equals, fn.get_file_of_type('image'), 'test.img.gz'
    yield assert_equals, fn.get_file_of_type('header'), 'test.img.hdr'
    # Now, tell the object to ignore these suffixes
    fn.ignored_suffixes=('.gz', '.bz2')
    fn.set_filenames('test.img.gz')
    yield assert_equals, fn.get_file_of_type('header'), 'test.hdr.gz'
    yield assert_equals, fn.get_file_of_type('image'), 'test.img.gz'
    fn.set_filenames('test.img.bz2')
    yield assert_equals, fn.get_file_of_type('header'), 'test.hdr.bz2'
    yield assert_equals, fn.get_file_of_type('image'), 'test.img.bz2'
    # but, just .gz or .bz2 as extension gives an error, if enforcing is on
    fn.enforce_extensions = True
    yield assert_raises, FileTuplesError, fn.set_filenames, 'test.gz'
    yield assert_raises, FileTuplesError, fn.set_filenames, 'test.bz2'
    # if enforcing is off, it tries to work out what the header should be
    # assuming the passed filename is the image
    fn.enforce_extensions = False
    fn.set_filenames('test.gz')
    yield assert_equals, fn.get_file_of_type('image'), 'test.gz'
    yield assert_equals, fn.get_file_of_type('header'), 'test.hdr.gz'
    # And not-recognized names also raise an error
    yield assert_raises, FileTuplesError, fn.get_file_of_type, 'obviously not present'
    # you can check if all the files are set with the 'all' call
    yield assert_true, fn.get_file_of_type('all'), ('test1.img', None)
    # Set each type separately with set_file_of_type
    fn.set_file_of_type('image', 'test1.img')
    fn.set_file_of_type('header', 'test2.hdr')
    yield assert_equals, fn.get_file_of_type('image'), 'test1.img'
    yield assert_equals, fn.get_file_of_type('header'), 'test2.hdr'
    # extensions are still enforced
    fn.enforce_extensions = True # the default    
    yield assert_raises, FileTuplesError, fn.set_file_of_type, 'image', 'test1.funny'
    # unless you tell the object not to
    fn.enforce_extensions = False
    fn.set_file_of_type('image', 'test1.funny')
    yield assert_equals, fn.get_file_of_type('image'), 'test1.funny'
    # You can put fileobjs into the individual slots with get_file_of_type
    fn = FileTuples((('image', '.img'), ('header', '.hdr')),
                   default_type = 'image',
                   enforce_extensions = True)
    s1 = StringIO()
    s2 = StringIO()
    fn.set_file_of_type('image', s1)
    fn.set_file_of_type('header', s2)
    yield assert_equals, fn.get_file_of_type('image'), s1
    yield assert_equals, fn.get_file_of_type('header'), s2
    # filenames remain unset, because - there are no filenames
    yield assert_equals, fn.get_filenames(), (None, None)
    # You obviously cannot use fileobjects for set_filenames
    # because the object can't work out the other names
    yield assert_raises, FileTuplesError, fn.set_filenames, s1
    # Setting file objects clears the relevant filename
    fn = FileTuples((('image', '.img'), ('header', '.hdr')),
                   default_type = 'image',
                   enforce_extensions = True)
    fn.set_filenames('test.img')
    yield assert_equals, fn.get_file_of_type('header'), 'test.hdr'
    fn.set_file_of_type('header', s2)
    yield assert_equals, fn.get_filenames(), ('test.img', None)

