from nibabel.orientations import aff2axcodes


class Field:
    """ Header fields common to multiple streamlines file formats.

    In IPython, use `nibabel.streamlines.Field??` to list them.
    """
    NB_STREAMLINES = "nb_streamlines"
    STEP_SIZE = "step_size"
    METHOD = "method"
    NB_SCALARS_PER_POINT = "nb_scalars_per_point"
    NB_PROPERTIES_PER_STREAMLINE = "nb_properties_per_streamline"
    NB_POINTS = "nb_points"
    VOXEL_SIZES = "voxel_sizes"
    DIMENSIONS = "dimensions"
    MAGIC_NUMBER = "magic_number"
    ORIGIN = "origin"
    VOXEL_TO_WORLD = "voxel_to_world"
    VOXEL_ORDER = "voxel_order"
    WORLD_ORDER = "world_order"
    ENDIAN = "endian"


def create_header_from_nifti(img):
    ''' Creates a common streamlines' header using a nifti image.

    Based on the information of the nifti image a dictionnary is created
    containing the following keys: `Field.ORIGIN`, `Field.DIMENSIONS`,
    `Field.VOXEL_SIZES`, `Field.VOXEL_TO_WORLD`, `Field.WORLD_ORDER`
    and `Field.VOXEL_ORDER`.

    Parameters
    ----------
    img : Nifti1Image object
        Image containing information about the anatomy where streamlines
        were created.

    Returns
    -------
    hdr : dict
        Header containing meta information about streamlines extracted
        from the anatomy.
    '''
    img_header = img.get_header()
    affine = img_header.get_best_affine()

    hdr = {}

    hdr[Field.ORIGIN] = affine[:3, -1]
    hdr[Field.DIMENSIONS] = img_header.get_data_shape()[:3]
    hdr[Field.VOXEL_SIZES] = img_header.get_zooms()[:3]
    hdr[Field.VOXEL_TO_WORLD] = affine
    hdr[Field.WORLD_ORDER] = "RAS"  # Nifti space
    hdr[Field.VOXEL_ORDER] = "".join(aff2axcodes(affine))

    return hdr
