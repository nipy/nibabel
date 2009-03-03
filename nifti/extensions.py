#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyNIfTI package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""This module provides a container-like interface to NIfTI1 header extensions.
"""

__docformat__ = 'restructuredtext'


# the NIfTI pieces
import nifti.clib as ncl

#
# type maps
#
nifti_ecode_map = \
    {"ignore": ncl.NIFTI_ECODE_IGNORE,
     "dicom": ncl.NIFTI_ECODE_DICOM,
     "afni": ncl.NIFTI_ECODE_AFNI,
     "comment": ncl.NIFTI_ECODE_COMMENT,
     "xcede": ncl.NIFTI_ECODE_XCEDE,
     "jimdiminfo": ncl.NIFTI_ECODE_JIMDIMINFO,
     "workflow_fwds": ncl.NIFTI_ECODE_WORKFLOW_FWDS,
     "freesurfer": ncl.NIFTI_ECODE_FREESURFER,
     # for now just conquer the next available ECODE
     "pypickle": 16,
     # later replace by the following when new niftilib is released
     #"pypickle": ncl.NIFTI_ECODE_PYPICKLE,
    }
nifti_ecode_inv_map = dict([(v, k) for k, v in nifti_ecode_map.iteritems()])


#
# little helpers
#
def _any2ecode(code, check_num=True):
    """Convert literal NIFTI_ECODEs into numerical ones.

    Both numerical and literal codes are check for validity by default. However,
    setting `check_num` to `False` will disable this check for numerical codes.
    """
    if isinstance(code, str):
        if not code in nifti_ecode_map.keys():
            raise ValueError, \
                  "Unknown ecode '%s'. Must be one of '%s'" \
                  % (code, str(nifti_ecode_map.keys()))
        code = nifti_ecode_map[code]
    elif check_num:
        if not code in nifti_ecode_map.values():
            raise ValueError, \
                  "Unknown ecode '%s'. Must be one of '%s'" \
                  % (str(code), str(nifti_ecode_map.values()))

    return code


#
# classes
#
class NiftiExtensions(object):
    """NIfTI1 header extension handler.

    This class wraps around a NIfTI1 struct and provides container-like access
    to NIfTI1 header extensions. It is basically a hibrid between a list and a
    dictionary. The reason for this is that the NIfTI header allows for a *list*
    of extensions, but additionally each extension is associated with some type
    (`ecode` or extension code). This is some form of *mapping*, however, the
    ecodes are not necessarily unique (e.g. multiple comments).

    The current list of known extensions is documented here:

      http://nifti.nimh.nih.gov/nifti-1/documentation/faq#Q21

    The usage is best explained by a few examples. All examples assume a NIfTI
    image to be loaded:

      >>> from nifti import NiftiImage
      >>> nim = NiftiImage('example4d.nii.gz')

    Access to the extensions is provided through the `extensions` attribute of
    the NiftiImage class:

      >>> type(nim.extensions)
      <class 'nifti.extensions.NiftiExtensions'>

    How many extensions are avialable?

      >>> len(nim.extensions)
      2

    How many comments? Any AFNI extension?

      >>> nim.extensions.count('comment')
      2

    Show me all `ecodes` of all extensions:

      >>> nim.extensions.ecodes
      [6, 6]

    Add an `AFNI` extension:

      >>> nim.extensions += ('afni', '<xml>Some voodoo</xml>')
      >>> nim.extensions.ecodes
      [6, 6, 4]

    Delete superfluous comment extension:

      >>> del nim.extensions[1]

    Access the last extension, which should be the `AFNI` one:

      >>> nim.extensions[-1]
      '<xml>Some voodoo</xml>'

    Wipe them all:

      >>> nim.extensions.clear()
      >>> len(nim.extensions)
      0

    """
    def __init__(self, raw_nimg, source=None):
        """
        :Parameters:
          raw_nimg: nifti_image struct
            This is the raw NIfTI image struct pointer. It is typically provided
            by :attr:`nifti.format.NiftiFormat.raw_nimg`.
          source: list(2-tuple)
            This is an optional list for extension tuples (ecode, edata). Each
            element of this list will be appended as a new extension.
        """
        # raw NIfTI image struct instance
        self.__raw_nimg = raw_nimg
        # wrapped extension list
        self.__rewrapExtList()

        if source:
            for ext in source:
                self.append(ext)


    def __rewrapExtList(self):
        """Grab a fresh pointer to the C extension array.
        """
        self.__elist = ncl.extensionArray_frompointer(self.__raw_nimg.ext_list)


    def __len__(self):
        return self.__raw_nimg.num_ext


    def __getitem__(self, key):
        # first try access by ascii ecode
        if isinstance(key, str):
            key = _any2ecode(key)

            # search for first matching ecode
            for c, d in self.iteritems():
                if c == key:
                    return d
        else:
            # support 'reverse' access by negative indices
            if key < 0:
                key += len(self)

            if key < len(self):
                return self.__elist[key].edata

        raise IndexError, \
              "Invalid extension ID '%s'." % `key`


    def __iter__(self):
        for i in xrange(len(self)):
            yield self[i]


    def iteritems(self):
        """A generator method that returns a 2-tuple (ecode, edata) on each
        iteration. It can be used in the same fashion as `dict.iteritems()`.
        """
        for i in xrange(len(self)):
            yield (self.__elist[i].ecode, self.__elist[i].edata)


    def __contains__(self, ecode):
        return _any2ecode(ecode) in [c for c, d in self.iteritems()]


    def count(self, code):
        """Returns the number of extensions matching a given *ecode*.

        :Parameter:
          code: int | str
            The ecode can be specified either literal or as numerical value.
        """
        # don't check numerical value to prevent unnecessary exceptions
        code = _any2ecode(code, check_num=False)
        count = 0
        for c, d in self.iteritems():
            if c == code:
                count += 1
        return count


    def ecodes(self):
        """Returns a list of ecodes for all extensions.
        """
        return [c for c, d in self.iteritems()]


    def append(self, extension):
        """Append a new extension.

        :Parameter:
          extension: 2-tuple
            An extension is given by a `(ecode, edata)` tuple, where `ecode` can
            be either literal or numerical and `edata` is any kind of data.

        Note:
          Currently, `edata` can only be stuff whos len(edata) matches its size
          in bytes, e.g. str.
        """
        ecode, edata = extension

        # check and potentially convert ecodes
        ecode = _any2ecode(ecode)

        # +1 to still include the stop bit if the extension length is n*16-8
        ret = ncl.nifti_add_extension(self.__raw_nimg,
                                      edata,
                                      len(edata) + 1,
                                      ecode)
        if not ret == 0:
            raise RuntimeError, \
                  "Could not add extension. Expect the world to end!"

        # make sure to rewrap the extension list to compensate for
        # changes the the C datatype/pointer madness
        self.__rewrapExtList()


    def __delitem__(self, key):
        # first try if we have an ascii ecode
        if isinstance(key, str):
            key = _any2ecode(key)
            key = self.ecodes.index(key)

        exts = [e for i, e in enumerate(self.iteritems()) if i != key]
        # tabula rasa
        self.clear()

        # and put remains extensions back
        for e in exts:
            self += e


    def clear(self):
        """Remove all extensions.
        """
        # tabula rasa
        ncl.nifti_free_extensions(self.__raw_nimg)


    def __del__(self):
        self.clear()


    def __iadd__(self, extension):
        self.append(extension)
        return self


    def __str__(self):
        return "extensions(%s)" % str([e for e in self.iteritems()])


    ecodes = property(fget=ecodes)
