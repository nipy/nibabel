class CoordinateImage:
    """
    Attributes
    ----------
    header : a file-specific header
    coordaxis : ``CoordinateAxis``
    dataobj : array-like
    """


class CoordinateAxis:
    """
    Attributes
    ----------
    parcels : list of ``Parcel`` objects
    """

    def load_structures(self, mapping):
        """
        Associate parcels to ``Pointset`` structures
        """
        raise NotImplementedError

    def __getitem__(self, slicer):
        """
        Return a sub-sampled CoordinateAxis containing structures
        matching the indices provided.
        """
        raise NotImplementedError

    def get_indices(self, parcel, indices=None):
        """
        Return the indices in the full axis that correspond to the
        requested parcel. If indices are provided, further subsample
        the requested parcel.
        """
        raise NotImplementedError


class Parcel:
    """
    Attributes
    ----------
    name : str
    structure : ``Pointset``
    indices : object that selects a subset of coordinates in structure
    """


class GeometryCollection:
    """
    Attributes
    ----------
    structures : dict
        Mapping from structure names to ``Pointset``
    """

    @classmethod
    def from_spec(klass, pathlike):
        """Load a collection of geometries from a specification."""
        raise NotImplementedError
