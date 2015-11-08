import numpy as np
import nibabel
import itertools

from nibabel.spatialimages import SpatialImage


def get_affine_from_reference(ref):
    """ Returns the affine defining the reference space.

    Parameter
    ---------
    ref : filename | `Nifti1Image` object | 2D array (4,4)
        Reference space where streamlines live in `fileobj`.

    Returns
    -------
    affine : 2D array (4,4)
    """
    if type(ref) is np.ndarray:
        if ref.shape != (4, 4):
            raise ValueError("`ref` needs to be a numpy array with shape (4,4)!")

        return ref
    elif isinstance(ref, SpatialImage):
        return ref.affine

    # Assume `ref` is the name of a neuroimaging file.
    return nibabel.load(ref).affine


def pop(iterable):
    "Returns the next item from the iterable else None"
    value = list(itertools.islice(iterable, 1))
    return value[0] if len(value) > 0 else None


def save_compact_list(filename, compact_list):
    np.savez(filename,
             data=compact_list._data,
             offsets=compact_list._offsets,
             lengths=compact_list._lengths)


def load_compact_list(filename):
    from .base_format import CompactList
    content = np.load(filename)
    compact_list = CompactList()
    compact_list._data = content["data"]
    compact_list._offsets = content["offsets"].tolist()
    compact_list._lengths = content["lengths"].tolist()
    return compact_list
