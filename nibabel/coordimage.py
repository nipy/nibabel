from nibabel.fileslice import fill_slicer


class CoordinateImage:
    """
    Attributes
    ----------
    header : a file-specific header
    coordaxis : ``CoordinateAxis``
    dataobj : array-like
    """

    def __init__(self, data, coordaxis, header=None):
        self.data = data
        self.coordaxis = coordaxis
        self.header = header


class CoordinateAxis:
    """
    Attributes
    ----------
    parcels : list of ``Parcel`` objects
    """

    def __init__(self, parcels):
        self.parcels = parcels

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
        if slicer is Ellipsis or slicer == slice(None):
            return self
        elif isinstance(slicer, slice):
            slicer = fill_slicer(slicer, len(self))
            start, stop, step = slicer.start, slicer.stop, slicer.step
        else:
            raise TypeError(f'Indexing type not supported: {type(slicer)}')

        subparcels = []
        pstop = 0
        for parcel in self.parcels:
            pstart, pstop = pstop, pstop + len(parcel)
            if pstop < start:
                continue
            if pstart >= stop:
                break
            if start < pstart:
                substart = (start - pstart) % step
            else:
                substart = start - pstart
            subparcels.append(parcel[substart : stop - pstart : step])
        return CoordinateAxis(subparcels)

    def get_indices(self, parcel, indices=None):
        """
        Return the indices in the full axis that correspond to the
        requested parcel. If indices are provided, further subsample
        the requested parcel.
        """
        raise NotImplementedError

    def __len__(self):
        return sum(len(parcel) for parcel in self.parcels)


class Parcel:
    """
    Attributes
    ----------
    name : str
    structure : ``Pointset``
    indices : object that selects a subset of coordinates in structure
    """

    def __init__(self, name, structure, indices):
        self.name = name
        self.structure = structure
        self.indices = indices

    def __repr__(self):
        return f'<Parcel {self.name}({len(self.indices)})>'

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, slicer):
        return self.__class__(self.name, self.structure, self.indices[slicer])


class GeometryCollection:
    """
    Attributes
    ----------
    structures : dict
        Mapping from structure names to ``Pointset``
    """

    def __init__(self, structures):
        self.structures = structures

    @classmethod
    def from_spec(klass, pathlike):
        """Load a collection of geometries from a specification."""
        raise NotImplementedError
