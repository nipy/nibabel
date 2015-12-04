from abc import ABCMeta, abstractmethod


class HeaderWarning(Warning):
    pass


class HeaderError(Exception):
    pass


class DataError(Exception):
    pass


class abstractclassmethod(classmethod):
    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)


class TractogramFile(object):
    ''' Convenience class to encapsulate tractogram file format. '''
    __metaclass__ = ABCMeta

    def __init__(self, tractogram, header=None):
        self._tractogram = tractogram
        self._header = {} if header is None else header

    @property
    def tractogram(self):
        return self._tractogram

    @property
    def streamlines(self):
        return self.tractogram.streamlines

    @property
    def header(self):
        return self._header

    def get_tractogram(self):
        return self.tractogram

    def get_streamlines(self):
        return self.streamlines

    def get_header(self):
        return self.header

    @classmethod
    def get_magic_number(cls):
        ''' Returns streamlines file's magic number. '''
        raise NotImplementedError()

    @classmethod
    def support_data_per_point(cls):
        ''' Tells if this tractogram format supports saving data per point. '''
        raise NotImplementedError()

    @classmethod
    def support_data_per_streamline(cls):
        ''' Tells if this tractogram format supports saving data per streamline. '''
        raise NotImplementedError()

    @classmethod
    def is_correct_format(cls, fileobj):
        ''' Checks if the file has the right streamlines file format.
        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to a streamlines file (and ready to read from the
            beginning of the header).
        Returns
        -------
        is_correct_format : boolean
            Returns True if `fileobj` is in the right streamlines file format.
        '''
        raise NotImplementedError()

    @abstractclassmethod
    def load(cls, fileobj, lazy_load=True):
        ''' Loads streamlines from a file-like object.
        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to a streamlines file (and ready to read from the
            beginning of the header).
        lazy_load : boolean (optional)
            Load streamlines in a lazy manner i.e. they will not be kept
            in memory. For postprocessing speed, turn off this option.
        Returns
        -------
        tractogram_file : ``TractogramFile`` object
            Returns an object containing tractogram data and header
            information.
        '''
        raise NotImplementedError()

    @abstractmethod
    def save(self, fileobj):
        ''' Saves streamlines to a file-like object.
        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            opened and ready to write.
        '''
        raise NotImplementedError()
