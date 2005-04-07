''' Create filename pairs, triplets etc, with expected extensions '''

import os

class TypesFilenamesError(Exception):
    pass


def types_filenames(template_fname, types_exts,
                    trailing_suffixes=('.gz', '.bz2'),
                    enforce_extensions=True):
    ''' Return filenames with standard extensions from template name

    The typical case is returning image and header filenames for an
    Analyze image, that expects and 'image' file type, with, extension
    ``.img``, and a 'header' file type, with extension ``.hdr``.

    Parameters
    ----------
    template_fname : str
       template filename from which to construct output dict of
       filenames, with given `types_exts` type to extension mapping.  If
       ``self.enforce_extensions`` is True, then filename must have one
       of the defined extensions from the types list.  If
       ``self.enforce_extensions`` is False, then the other filenames
       are guessed at by adding extensions to the base filename.
       Ignored suffixes (from `trailing_suffixes`) append themselves to
       the end of all the filenames.
    types_exts : sequence of sequences
       sequence of (name, extension) str sequences defining type to
       extension mapping.
    trailing_suffixes : sequence of strings, optional
        suffixes that should be ignored when looking for
        extensions - default is ``('.gz', '.bz2')``
    enforce_extensions : {True, False}, optional
        If True, raise an error when attempting to set value to
        type which has the wrong extension

    Returns
    -------
    types_fnames : dict
       dict with types as keys, and generated filenames as values.  The
       types are given by the first elements of the tuples in
       `types_exts`. 

    Examples
    --------
    >>> types_exts = (('t1','.ext1'),('t2', '.ext2'))
    >>> tfns = types_filenames('/path/test.ext1', types_exts)
    >>> tfns == {'t1': '/path/test.ext1', 't2': '/path/test.ext2'}
    True

    Bare file roots without extensions get them added
    
    >>> tfns = types_filenames('/path/test', types_exts)
    >>> tfns == {'t1': '/path/test.ext1', 't2': '/path/test.ext2'}
    True

    With enforce_extensions == False, allow first type to have any
    extension.
    
    >>> tfns = types_filenames('/path/test.funny', types_exts,
    ...                        enforce_extensions=False)
    >>> tfns == {'t1': '/path/test.funny', 't2': '/path/test.ext2'}
    True
    '''
    if not isinstance(template_fname, basestring):
        raise TypesFilenamesError('Need file name as input to set_filenames')
    if template_fname.endswith('.'):
        template_fname = template_fname[:-1]
    filename, found_ext, ignored, guessed_name = \
              _parse_filename(template_fname, types_exts, trailing_suffixes)
    # Flag cases where we just set the input name directly
    direct_set_name = None
    if enforce_extensions:
        if guessed_name is None:
            # no match - maybe there was no extension atall or the
            # wrong extension
            if found_ext:
                # an extension, but the wrong one
                raise TypesFilenamesError(
                    'File extension "%s" was not in expected list: %s'
                    % (found_ext, [e for t, e in types_exts]))
            elif ignored: # there was no extension, but an ignored suffix
                # This is a special case like 'test.gz' (where .gz
                # is ignored). It's confusing to change
                # this to test.img.gz, or test.gz.img, so error
                raise TypesFilenamesError(
                    'Confusing ignored suffix %s without extension'
                    % ignored)
    else: # not enforcing extensions
        # if there's an extension, we set the filename directly
        # from input.  Also, if there was no extension, but an
        # ignored suffix ('test.gz' type case), we set the
        # filename directly.  Otherwise (no extension, no ignored
        # suffix), we stay with the default, which is to add the
        # default extensions according to type.
        if found_ext or ignored:
            direct_set_name = types_exts[0][0]
    tfns = {}
    for name, ext in types_exts:
        if name == direct_set_name:
            tfns[name] = template_fname
            continue
        fname = filename
        if ext:
            fname += ext
        if ignored:
            fname += ignored
        tfns[name] = fname
    return tfns


def _parse_filename(filename, types_exts, trailing_suffixes):
    ''' Splits filename into tuple of
    (fileroot, extension, trailing_suffix, guessed_name)

    >>> types_exts = (('t1', 'ext1'),('t2', 'ext2'))
    >>> _parse_filename('/path/fname.funny', types_exts, ())
    ('/path/fname', '.funny', None, None)
    >>> _parse_filename('/path/fnameext2', types_exts, ())
    ('/path/fname', 'ext2', None, 't2')
    >>> _parse_filename('/path/fnameext2', types_exts, ('.gz',))
    ('/path/fname', 'ext2', None, 't2')
    >>> _parse_filename('/path/fnameext2.gz', types_exts, ('.gz',))
    ('/path/fname', 'ext2', '.gz', 't2')
    '''
    ignored = None
    for ext in trailing_suffixes:
        if filename.endswith(ext):
            filename = filename[:-(len(ext))]
            ignored = ext
            break
    guessed_name = None
    found_ext = None
    tem = dict(types_exts)
    for name, ext in types_exts:
        if ext and filename.endswith(ext):
            filename = filename[:-(len(ext))]
            guessed_name = name
            found_ext = ext
            break
    else:
        filename, found_ext = os.path.splitext(filename)
    return (filename, found_ext, ignored, guessed_name)


def splitext_addext(filename, addexts=('.gz', '.bz2')):
    ''' Split ``/pth/fname.ext.gz`` into ``/pth/fname, .ext, .gz``

    where ``.gz`` may be any of passed `addext` trailing suffixes.

    Parameters
    ----------
    filename : str
       filename that may end in any or none of `addexts`

    Returns
    -------
    froot : str
       Root of filename - e.g. ``/pth/fname`` in example above
    ext : str
       Extension, where extension is not in `addexts` - e.g. ``.ext`` in
       example above
    addext : str
       Any suffixes appearing in `addext` occuring at end of filename

    Examples
    --------
    >>> splitext_addext('fname.ext.gz')
    ('fname', '.ext', '.gz')
    >>> splitext_addext('fname.ext')
    ('fname', '.ext', '')
    >>> splitext_addext('fname.ext.foo', ('.foo', '.bar'))
    ('fname', '.ext', '.foo')
    ''' 
    for ext in addexts:
        if filename.endswith(ext):
            filename = filename[:-(len(ext))]
            addext = ext
            break
    else:
        addext = ''
    return os.path.splitext(filename) + (addext,)
