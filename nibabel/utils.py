""" Code support routines, not otherwise classified
"""


def to_scalar(val):
    """ Return scalar representation of `val`

    Return scalar value from numpy array, or pass through value if not numpy
    array.

    Parameters
    ----------
    val : object
        numpy array or other object.

    Returns
    -------
    out : object
        Result of ``val.item()`` if `val` has an ``item`` method, else `val`.
    """
    return val.item() if hasattr(val, 'item') else val
