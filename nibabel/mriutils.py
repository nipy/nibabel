# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities for calculations related to MRI
"""

__all__ = ['calculate_dwell_time']

GYROMAGNETIC_RATIO = 42.576  # MHz/T for nucleus
PROTON_WATER_FAT_SHIFT = 3.4  # ppm


def calculate_dwell_time(water_fat_shift, echo_train_length,
                         field_strength=3.0):
    """Calculate the dwell time

    Parameters
    ----------
    water_fat_shift : float
        The water fat shift of the recording, in pixels.
    echo_train_length : int
        The echo train length of the imaging sequence.
    field_strength : float
        Strength of the magnet in T, e.g. ``3.0`` for a 3T magnet
        recording. Providing this value is necessary because the
        field strength is not encoded in the PAR file.

    Returns
    -------
    dwell_time : float
        The dwell time in seconds. Returns None if the dwell
        time cannot be calculated (i.e., not using an EPI sequence).
    """
    field_strength = float(field_strength)  # Tesla
    assert field_strength > 0.
    if echo_train_length <= 0:
        return None
    # constants
    dwell_time = ((echo_train_length - 1) * water_fat_shift /
                  (GYROMAGNETIC_RATIO * PROTON_WATER_FAT_SHIFT
                   * field_strength * (echo_train_length + 1)))
    return dwell_time
