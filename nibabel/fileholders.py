''' Fileholder class '''

from nibabel.volumeutils import allopen


class FileHolderError(Exception):
    pass


class FileHolder(object):
    ''' class to contain filename, fileobj and file position
    '''
    def __init__(self,
                 filename=None,
                 fileobj=None,
                 pos=0):
        ''' Initialize FileHolder instance

        Parameters
        ----------
        filename : str, optional
           filename.  Default is None
        fileobj : file-like object, optional
           Should implement at least 'seek' (for the purposes for this
           class).  Default is None
        pos : int, optional
           position in filename or fileobject at which to start reading
           data; defaults to 0
        '''
        self.filename = filename
        self.fileobj = fileobj
        self.pos=pos

    def has_file(self):
        ''' Return True if filename or fileobj are set '''
        return self.filename is not None or self.fileobj is not None

    def get_prepare_fileobj(self, *args, **kwargs):
        ''' Return fileobj if present, or return fileobj from filename

        Set position to that given in self.pos

        Parameters
        ----------
        *args : tuple
           positional arguments to file open.  Ignored if there is a
           defined ``self.fileobj``.  These might include the mode, such
           as 'rb'
        **kwargs : dict
           named arguments to file open.  Ignored if there is a
           defined ``self.fileobj``

        Returns
        -------
        fileobj : file-like object
           object has position set (via ``fileobj.seek()``) to
           ``self.pos``
        '''
        if self.fileobj is not None:
            obj = self.fileobj
            obj.seek(self.pos)
        elif self.filename is not None:
            obj = allopen(self.filename, *args, **kwargs)
            if self.pos != 0:
                obj.seek(self.pos)
        else:
            raise FileHolderError('No filename or fileobj present')
        return obj
