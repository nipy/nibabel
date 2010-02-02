# module imports
from nibabel import volumeutils as vu
from nibabel import spm2analyze as spm2
from nibabel import nifti1
from nibabel import minc


def load(filename, *args, **kwargs):
    ''' Load file given filename, guessing at file type

    Parameters
    ----------
    filename : string or file-like
       specification of filename or file to load
    *args
    **kwargs
       arguments to pass to image load function

    Returns
    -------
    img : ``SpatialImage``
       Image of guessed type

    '''
    # Try and guess file type from filename
    if isinstance(filename, basestring):
        fname = filename
        for ending in ('.gz', '.bz2'):
            if filename.endswith(ending):
                fname = fname[:-len(ending)]
                break
        if fname.endswith('.nii'):
            return nifti1.load(filename, *args, **kwargs)
        if fname.endswith('.mnc'):
            return minc.load(filename, *args, **kwargs)
    # Not a string, or not recognized as nii or mnc
    try:
        files = nifti1.Nifti1Image.filespec_to_files(filename)
    except ValueError:
        raise RuntimeError('Cannot work out file type of "%s"' %
                           filename)
    hdr = nifti1.Nifti1Header.from_fileobj(
        vu.allopen(files['header']),
        check=False)
    magic = hdr['magic']
    if magic in ('ni1', 'n+1'):
        return nifti1.load(filename, *args, **kwargs)
    return spm2.load(filename, *args, **kwargs)


def save(img, filename):
    ''' Save an image to file without changing format'''
    img.to_filename(filename)
