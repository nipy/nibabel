#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
''' Base class for images

A draft.
'''

import nifti.ioimps as ioimps

class ImageError(Exception):
    pass

class Image(object):
    ''' Base class for images

    Attributes:

    * data : array-like; image data
    * affine : array; mapping from image voxels to output space
      coordinates
    * output_space : string; name for affine output space
    * meta : dict-like; image metadata
    * io : image io implementation object

    Properties (sorry guys):

    * filename : string : read only, filename or None if none
    * mode : string; 'r'ead, 'w'rite, or 'rw' 
    
    Methods:

    * save(filespec=None)
    * get_filespec()
    * set_filespec(filespec) 
    * is_file()

    Class attributes:

    * default_io_class

    '''

    default_io_class = ioimps.default_io
    
    def __init__(self, data,
                 affine=None,
                 output_space=None,
                 meta=None,
                 filespec=None,
                 io=None,
                 mode='rw'):
        self.data = data
        self.affine = affine
        self.output_space = output_space
        if meta is None:
            meta = {}
        self.meta = meta
        if not filespec is None:
            if io is None:
                io = ioimps.guessed_imp(filespec)
            else:
                io.set_filespec(filespec)
        if io is None:
            io = self.default_io_class()
        self.io = io
        self._mode = None
        self.mode = mode
        
    @property
    def filename(self):
        filespec = self.get_filespec()
        return filespec['image']

    def mode():
        def fget(self):
            return self._mode
        def fset(self, mode):
            if not set('rw').issuperset(set(mode)):
                raise ImageError('Invalid mode "%s"' % mode)
            self._mode = mode
        doc = 'image read / write mode'
        return locals()
    mode = property(**mode())
            
    def get_filespec(self):
        return self.io.get_filespec()

    def set_filespec(self, filespec):
        self.io.set_filespec(filespec)
        
    def save(self, filespec=None, io=None):
        if filespec is None:
            filespec = self.filespec
        if io is None:
            io = self.io
        return io.to_filespec(filespec)

    def from_io(klass, io):
        io = io.copy()
        data = io.get_data_proxy()
        affine = io.get_affine()
        output_space = io.get_output_space()
        return klass(data,
                     affine,
                     output_space,
                     {},
                     filespec = None,
                     io=io)
                     
    def is_file(self):
        ''' True if image in memory is same as image on disk '''
        return False # we don't handle this yet
   

def load(filespec, maker=Image, io=None):
    if io is None:
        io = ioimps.guessed_implementation(filespec)
    else:
        io.set_filespec(filespec)
    return maker.from_io(io)


def save(img, filespec=None, io=None):
    return img.save(filespec, io)
