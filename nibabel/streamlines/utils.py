import os

import nibabel as nib

from nibabel.streamlines.trk import TrkFile
#from nibabel.streamlines.tck import TckFile
#from nibabel.streamlines.vtk import VtkFile
#from nibabel.streamlines.fib import FibFile

# Supported format
FORMATS = {"trk": TrkFile,
           #"tck": TckFile,
           #"vtk": VtkFile,
           #"fib": FibFile,
           }

def is_supported(fileobj):
    return detect_format(fileobj) is not None


def detect_format(fileobj):
    for format in FORMATS.values():
        if format.is_correct_format(fileobj):
            return format

    if isinstance(fileobj, basestring):
        _, ext = os.path.splitext(fileobj)
        return FORMATS.get(ext, None)

    return None


def convert(in_fileobj, out_filename):
    in_fileobj = nib.streamlines.load(in_fileobj)
    out_format = nib.streamlines.guess_format(out_filename)

    hdr = in_fileobj.get_header()
    points = in_fileobj.get_points(as_generator=True)
    scalars = in_fileobj.get_scalars(as_generator=True)
    properties = in_fileobj.get_properties(as_generator=True)

    out_fileobj = out_format(hdr, points, scalars, properties)
    out_fileobj.save(out_filename)


def change_space(streamline_file, new_point_space):
    pass