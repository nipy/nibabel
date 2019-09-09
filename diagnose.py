#!/usr/bin/env python3
import numpy as np
import nibabel as nb
from pprint import pprint

def _repr(obj):
    if isinstance(obj, dict):
        return {k: _repr(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return obj.__class__(_repr(v) for v in obj)

    if 'at 0x' in repr(obj):
        return repr(obj)

    return "<repr:{obj!r}; str:{obj!s} at 0x{objid:x}>".format(obj=obj, objid=id(obj))

pprint(nb.pkg_info.get_pkg_info('nibabel'))

pprint(_repr(np.sctypes), width=100)

for fp in np.sctypes['float']:
    print((_repr(np.dtype(fp)), _repr(fp)))
