import argparse
import nibabel as nb


def lossless_slice(img, slicers):
    if not nb.imageclasses.spatial_axes_first(img):
        raise ValueError("Cannot slice an image that is not known to have spatial axes first")

    roi_img = img.__class__(
        img.dataobj._get_unscaled(slicers),
        affine=img.slicer.slice_affine(slicers),
        header=img.header)
    roi_img.header.set_slope_inter(img.dataobj.slope, img.dataobj.inter)
    return roi_img


def parse_slice(crop, allow_step=True):
    if crop is None:
        return slice(None)
    start, stop, *extra = [int(val) if val else None for val in crop.split(":")]
    if len(extra) > 1:
        raise ValueError(f"Cannot parse specification: {crop}")
    if extra and not allow_step:
        raise ValueError(f"Step entry not permitted: {crop}")

    step = extra[0] if extra else None
    if step not in (1, -1, None):
        raise ValueError(f"Downsampling is not supported: {crop}")

    return slice(start, stop, step)


def main():
    parser = argparse.ArgumentParser(description="Crop images to a region of interest",
                                     epilog="If a start or stop value is omitted, the start or end of the axis is assumed.")
    parser.add_argument("-i", metavar="I1:I2[:-1]",
                        help="Start/stop [flip] along first axis (0-indexed)")
    parser.add_argument("-j", metavar="J1:J2[:-1]",
                        help="Start/stop [flip] along second axis (0-indexed)")
    parser.add_argument("-k", metavar="K1:K2[:-1]",
                        help="Start/stop [flip] along third axis (0-indexed)")
    parser.add_argument("-t", metavar="T1:T2", help="Start/stop along fourth axis (0-indexed)")
    parser.add_argument("in_file", help="Image file to crop")
    parser.add_argument("out_file", help="Output file name")

    opts = parser.parse_args()

    try:
        islice = parse_slice(opts.i)
        jslice = parse_slice(opts.j)
        kslice = parse_slice(opts.k)
        tslice = parse_slice(opts.t, allow_step=False)
    except ValueError as err:
        print(f"Could not parse input arguments. Reason follows.\n{err}")
        return 1

    img = nb.load(opts.in_file)
    try:
        sliced_img = lossless_slice(img, (islice, jslice, kslice, tslice)[:img.ndim])
    except:
        print("Could not slice image. Full traceback follows.")
        raise
    nb.save(sliced_img, opts.out_file)
    return 0
