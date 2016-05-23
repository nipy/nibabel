import numpy as np
import nibabel


def get_affine_from_reference(ref):
    """ Returns the affine defining the reference space.

    Parameter
    ---------
    ref : str or :class:`Nifti1Image` object or ndarray shape (4, 4)
        If str then it's the filename of reference file that will be loaded
        using :func:nibabel.load in order to obtain the affine.
        If :class:`Nifti1Image` object then the affine is obtained from it.
        If ndarray shape (4, 4) then it's the affine.

    Returns
    -------
    affine : ndarray (4, 4)
        Transformation matrix mapping voxel space to RAS+mm space.
    """
    if type(ref) is np.ndarray:
        if ref.shape != (4, 4):
            msg = "`ref` needs to be a numpy array with shape (4, 4)!"
            raise ValueError(msg)

        return ref
    elif hasattr(ref, 'affine'):
        return ref.affine

    # Assume `ref` is the name of a neuroimaging file.
    return nibabel.load(ref).affine
