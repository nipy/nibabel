"""Reading functions for freesurfer files
"""

from io import read_geometry, read_morph_data, \
               read_annot, read_label
from mghformat import load, save, MGHImage
