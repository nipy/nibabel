""" Testing package info
"""

import nibabel as nib

def test_pkg_info():
    """Simple smoke test
    
    Hits:
        - nibabel.get_info
        - nibabel.pkg_info.get_pkg_info
        - nibabel.pkg_info.pkg_commit_hash
    """
    info = nib.get_info()
