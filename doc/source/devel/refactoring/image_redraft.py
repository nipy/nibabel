import numpy as np

def _unwritable_wrapper(in_arr):
     out_arr = np.asanyarray(arr)
     if not out_arr.flags.writeable:
          return out_arr
     if out_arr is arr: # there was an array as input
          out_arr = arr[:]
     out_arr.flags.writeable = False
     return out_arr


class Image(object):
     _header_class = Header # generic Header class
     
     def __init__(self, data, affine, header=None, world='unknown', extra=None):
          self._data = _unwritable_wrapper(data)
          self.set_affine(affine)
          self.set_header(header)
          self._world = world
          if extra is None:
               extra = {}
          self.extra = extra

     def get_data(self, copy=False):
          data = self._data
          if copy:
               return data.copy()
          return data

     def get_affine(self):
          return self._affine.copy()

     def set_affine(self, affine):
          self._affine = np.array(affine) # copy
          self._dirty_header = True

     def get_header(self):
          return self._header.copy()

     def set_header(self, header):
          self._header = self._header_class.from_header(header) # copy
          self._dirty_header = True

     def _get_world(self):
          return self._world

     def _set_world(self, world):
          self._world = world
          self._dirty_header = True

     world = property(_get_world, _set_world, None, 'Get / set world')
