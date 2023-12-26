"""Reading functions for freesurfer files
"""

# ruff: noqa: F401

from .io import (
    read_annot,
    read_geometry,
    read_label,
    read_morph_data,
    write_annot,
    write_geometry,
    write_morph_data,
)
from .mghformat import MGHImage, load, save
