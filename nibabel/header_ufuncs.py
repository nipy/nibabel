'''  Ufunc-like functions operating on Analyze headers '''
import numpy as np

from nibabel.volumeutils import array_from_file, array_to_file

from nibabel.spatialimages import HeaderDataError


def read_data(hdr, fileobj):
    ''' Read data from ``fileobj`` given ``hdr``

    Parameters
    ----------
    hdr : header
       analyze-like header implementing ``get_slope_inter`` and
       requirements for ``read_unscaled_data``
    fileobj : file-like
       Must be open, and implement ``read`` and ``seek`` methods

    Returns
    -------
    arr : array-like
       an array like object (that might be an ndarray),
       implementing at least slicing.

    '''
    # read unscaled data
    dtype = hdr.get_data_dtype()
    shape = hdr.get_data_shape()
    offset = hdr.get_data_offset()
    data = array_from_file(shape, dtype, fileobj, offset)
    # get scalings from header
    slope, inter = hdr.get_slope_inter()
    if slope is None:
        return data
    # The data may be from a memmap, and not writeable
    if slope:
        if slope !=1.0:
            try:
                data *= slope
            except ValueError:
                data = data * slope
        if inter:
            try:
                data += inter
            except ValueError:
                data = data + inter
    return data


def write_scaled_data(hdr, data, fileobj):
    ''' Write data to ``fileobj`` with best data match to ``hdr`` dtype

    This is a convenience function that modifies the header as well as
    writing the data to file.  Because it modifies the header, it is not
    very useful for general image writing, where you often need to first
    write the header, then the image.

    Parameters
    ----------
    hdr : analyze-type header
       implemting ``get_data_shape``
    data : array-like
       data to write; should match header defined shape
    fileobj : file-like object
       Object with file interface, implementing ``write`` and ``seek``

    Returns
    -------
    None

    Examples
    --------
    >>> from nibabel.analyze import AnalyzeHeader
    >>> hdr = AnalyzeHeader()
    >>> hdr.set_data_shape((1, 2, 3))
    >>> hdr.set_data_dtype(np.float64)
    >>> from StringIO import StringIO
    >>> str_io = StringIO()
    >>> data = np.arange(6).reshape(1,2,3)
    >>> write_scaled_data(hdr, data, str_io)
    >>> data.astype(np.float64).tostring('F') == str_io.getvalue()
    True
    '''
    data = np.asarray(data)
    slope, inter, mn, mx = hdr.scaling_from_data(data)
    shape = hdr.get_data_shape()
    if data.shape != shape:
        raise HeaderDataError('Data should be shape (%s)' %
                              ', '.join(str(s) for s in shape))
    offset = hdr.get_data_offset()
    out_dtype = hdr.get_data_dtype()
    array_to_file(data, fileobj, out_dtype, offset, inter, slope, mn, mx)
    hdr.set_slope_inter(slope, inter)
