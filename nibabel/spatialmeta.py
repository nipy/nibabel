# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
SpatialMeta class for tracking potentially varying meta data associated 
with a spatial image. Can be merged or split along any dimension while 
maintaining the correct set of meta data.
"""
from __future__ import division

import sys, json, warnings, itertools, re
from copy import deepcopy
from collections import Mapping, Container
from itertools import combinations, product
import numpy as np

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

try:
    basestring
except NameError:  # python 3
    basestring = str

try:
    range = xrange
except NameError:  # python 3
    pass

def is_constant(sequence, period=None):
    '''Returns true if all elements in (each period of) the sequence are 
    equal. 
    
    Parameters
    ----------
    sequence : sequence
        The sequence of elements to check.
    
    period : int
        If not None then each subsequence of that length is checked. 
    '''
    if period is None:
        return all(val == sequence[0] for val in sequence)
    else:
        if period <= 1:
            raise ValueError('The period must be greater than one')
        seq_len = len(sequence)
        if seq_len % period != 0:
            raise ValueError('The sequence length is not evenly '
                             'divisible by the period length.')
                             
        for period_idx in range(seq_len // period):
            start_idx = period_idx * period
            end_idx = start_idx + period
            if not all(val == sequence[start_idx] 
                       for val in sequence[start_idx:end_idx]):
                return False
    
    return True
    
def is_repeating(sequence, period):
    '''Returns true if the elements in the sequence repeat with the given 
    period.

    Parameters
    ----------
    sequence : sequence
        The sequence of elements to check.
        
    period : int
        The period over which the elements should repeat.    
    '''
    seq_len = len(sequence)
    if period <= 1 or period >= seq_len:
        raise ValueError('The period must be greater than one and less than '
                         'the length of the sequence')
    if seq_len % period != 0:
        raise ValueError('The sequence length is not evenly divisible by the '
                         'period length.')
                         
    for period_idx in range(1, seq_len // period):
        start_idx = period_idx * period
        end_idx = start_idx + period
        if sequence[start_idx:end_idx] != sequence[:period]:
            return False
    
    return True

_meta_version = 0.7
'''Current meta data version'''

_req_base_keys_map= {0.5 : set(('dcmmeta_affine', 
                                'dcmmeta_slice_dim',
                                'dcmmeta_shape',
                                'dcmmeta_version',
                                'global',
                                )
                               ),
                     0.6 : set(('dcmmeta_affine', 
                                'dcmmeta_reorient_transform',
                                'dcmmeta_slice_dim',
                                'dcmmeta_shape',
                                'dcmmeta_version',
                                'global',
                                )
                               ),
                     0.7 : set(('image_meta',
                                'version',
                                'varies_over()',
                               )
                              )
                    }
'''Maps version numbers to minimum required keys in the base dictionary 
to be considered valid'''

class InvalidSpatialMetaError(Exception):
    def __init__(self, msg):
        '''Exception denoting that a SpatialMeta object is invalid.'''
        self.msg = msg
    
    def __str__(self):
        return 'The SpatialMeta is not valid: %s' % self.msg
        
class SpatialMeta(object):
    '''Nested mapping for storing meta data associated with a SpatialImage. 
    The first three dimensions of the image are considered the spatial 
    dimensions, and any further dimensions are considered non-spatial.
    
    The meta data can vary over the dimensions of the image. Each meta data 
    element is classified based on if, and how, it repeats over the image 
    dimensions. Each classification is stored in a nested dictionary.
    
    The meta data itself does not need to be flat, it can be nested into 
    mappings. The leafs of these nested mappings are considered as individual 
    elements. Each of these elements has a unique classification, and can be 
    referred to with a tuple of keys (giving the key for each level of 
    nesting).
    '''
    
    def __init__(self, shape, affine):
        '''Make an empty SpatialMeta object.
        
        Parameters
        ----------
        shape : tuple
            The shape of the image associated with this extension.
            
        affine : array
            The RAS affine for the image associated with this extension.
        '''
        self._data = OrderedDict()

        #Create nested dict for storing image meta data 
        self._data['image_meta'] = OrderedDict()
        
        #Set all of the image meta data
        self.shape = shape
        self.affine = affine
        self.version = _meta_version
        
        #Create a nested dict for each classification of meta data
        for classification in self.get_classes():
            self._data[classification] = OrderedDict()
    
    @property
    def affine(self):
        '''The affine for the image associated with this SpatialMeta object.'''
        return np.array(self._data['image_meta']['affine'])
        
    @affine.setter
    def affine(self, value):
        if value.shape != (4, 4):
            raise ValueError("Invalid shape for affine")
        self._data['image_meta']['affine'] = value.tolist()
    
    @property
    def shape(self):
        '''The shape of the image associated with the meta data. Defines the 
        number of values for the meta data classifications.'''
        return tuple(self._data['image_meta']['shape'])
    
    @shape.setter
    def shape(self, value):
        if len(value) < 2:
            raise ValueError("There must be at least two dimensions")
        if len(value) == 2:
            value = value + (1,)
        self._data['image_meta']['shape'] = value
            
    @property
    def version(self):
        '''The version of the meta data format.'''
        return self._data['version']
        
    @version.setter
    def version(self, value):
        '''Set the version of the meta data format.'''
        self._data['version'] = value
    
    def get_classes(self):
        '''Return the meta data classifications that are available.
        
        Returns
        -------
        classes : tuple
            The classifications that are available for this object (based 
            on the shape).
        
        '''
        shape = self.shape
        n_dims = len(shape)
        
        #Classification for constant meta data
        result = ['varies_over()']

        #Add classification for each individual (non-singular) dimension
        for dim_idx in range(n_dims):
            if shape[dim_idx] != 1:
                result.append('varies_over(%d)' % dim_idx)
            
        #Find number of non-spatial dimensions
        n_non_spatial = 0
        ns_dim_idxs = []
        for dim_idx in range(3, n_dims):
            if shape[dim_idx] != 1:
                n_non_spatial += 1
                ns_dim_idxs.append(dim_idx)
                
        #Elements can vary over any combination of non-spatial dimensions
        for n_ns_dim in range(2, n_non_spatial + 1):
            for ns_dims in combinations(ns_dim_idxs, n_ns_dim):
                result.append('varies_over(' + 
                              ','.join(str(x) for x in ns_dims) + ')'
                             )
            
        #Elements can vary over a combination of one spatial dimension and any 
        #number of non-spatial dimensions
        for sp_dim in range(3):
            for n_ns_dim in range(1, n_non_spatial + 1):
                for ns_dims in combinations(ns_dim_idxs, n_ns_dim):
                    result.append('varies_over(' + 
                                  ','.join(str(x) 
                                           for x in ((sp_dim,) + ns_dims)) + 
                                  ')'
                                 )
                                 
        return tuple(result)
    
    @classmethod
    def get_varying_dims(klass, classification):
        '''Return list of dimensions this classification varies over.'''
        match = re.match('varies_over\((.*)\)', classification)
        if not match:
            raise ValueError("Invalid classification")
        val_str = match.groups()[0]
        if val_str == '':
            return []
        else:
            return [int(tok.strip()) for tok in val_str.split(',')]
            
    def get_n_vals(self, classification):
        '''Get the number of values for all meta data elements of the 
        provided classification.
        
        Parameters
        ----------
        classification : tuple
            The meta data classification.
            
        Returns
        -------
        n_vals : int
            The number of values for any meta data of the provided 
            `classification`.
        '''
        if not classification in self.get_classes():
            raise ValueError("Invalid classification: %s" % classification)
        
        shape = self.shape
        varying_dims = self.get_varying_dims(classification)
        
        n_vals = 1
        for dim_idx in varying_dims:
            n_vals *= shape[dim_idx]
        
        return n_vals
    
    def get_nested(self, classification, keys):
        '''Get nested value with given classification and keys.
        
        Parameters
        ----------
        classification : str
            The classification of the meta data
        
        keys : iterable
            The keys to use at each level of nesting.
        '''
        curr_elem = self._data[classification]
        for key in keys:
            #Only look into nested mappings
            if not isinstance(curr_elem, Mapping):
                raise KeyError("Keys not found: %s" % (keys,))
            curr_elem = curr_elem[key]
        return curr_elem
            
    def set_nested(self, classification, keys, val):
        '''Set the value within a nested dict, creating any non-existant 
        intermediate dicts along the way
        
        Parameters
        ----------
        classification : str
            The classification of the meta data
        
        keys : tuple
            The keys to use at each level of nesting
            
        val : object
            The value to set for the leaf
        '''
        curr_elem = self._data[classification]
        for key in keys[:-1]:
            if not key in curr_elem:
                curr_elem[key] = OrderedDict()
            curr_elem = curr_elem[key]
        curr_elem[keys[-1]] = val
        
    def contains_nested(self, classification, keys):
        '''Returns true if the nested element exists, else false.'''
        curr_elem = self._data[classification]
        for key in keys:
            if not key in curr_elem:
                return False
            curr_elem = curr_elem[key]
        return True
        
    def remove_nested(self, classification, keys):
        '''Remove a nested element, deleting any intermediate dictionaries 
        that become empty.'''
        curr_dict = self._data[classification]
        prev_dicts = []
        for sub_key in keys:
            if not sub_key in curr_dict:
                break
            prev_dicts.append(curr_dict)
            curr_dict = curr_dict[sub_key]
        else:
            for depth_idx in range(len(keys)-1, -1, -1):
                del prev_dicts[depth_idx][keys[depth_idx]]
                if len(prev_dicts[depth_idx]) != 0:
                    break
            return
        raise KeyError("Keys not found: %s" % (keys,))
            
    def iter_elements(self, classification):
        '''Generates the key tuple and value for each meta data element of the 
        given classification. The leaf nodes of any nested dictionaries 
        are considered the individual meta data elements. The key tuple 
        contains a key for each level the meta data is nested.
        
        Parameters
        ----------
        classification : str
            The classification to generate elements from.
        '''
        valid_classes = self.get_classes()
        if not classification in valid_classes:
            raise ValueError("Not a valid classification: %s" % classification)
        
        iters = [(tuple(), self._data[classification].iteritems())]
        while iters:
            key_prefix, it = iters.pop()
            for k, v in it:
                new_prefix = (key_prefix + (k,))
                if isinstance(v, Mapping):
                    iters.append((new_prefix, v.iteritems()))
                else:
                    yield (new_prefix, v)
    
    def check_valid(self):
        '''Check if the extension is valid.
        
        Raises
        ------
        InvalidSpatialMetaError 
            The extension is missing required meta data or classifications, or
            some element(s) have the wrong number of values for their 
            classification.
        '''
        #TODO: Update meta data from older versions
        #For now we just fail out with older versions of the meta data
        if self.version != _meta_version:
            raise InvalidSpatialMetaError("Meta data version is out of date, "
                                          "you may be able to convert to the "
                                          "current version.")
                                        
        #Check for the required base keys in the json data
        if not _req_base_keys_map[self.version] <= set(self._data):
            raise InvalidSpatialMetaError('Missing one or more required keys')
            
        #Check we have a dictionary for all valid classifications, and that 
        #each element has the correct number of values and a unique 
        #classification
        valid_classes = self.get_classes()
        for classification in valid_classes:
            if not classification in self._data:
                raise InvalidSpatialMetaError('Missing required classification ' 
                                              '%s' % classification)
            
            curr_n_vals = self.get_n_vals(classification)
            for key_tup, vals in self.iter_elements(classification):
                if curr_n_vals != 1:
                    n_vals = len(vals)
                    if n_vals != curr_n_vals:
                        msg = (('Incorrect number of values for element %s '
                                'with classification %s, expected %d found %d'
                               ) %
                               (key_tup, classification, curr_n_vals, n_vals)
                              )
                        raise InvalidSpatialMetaError(msg)
                if any(self.contains_nested(c, key_tup) 
                       for c in valid_classes 
                       if not c == classification
                      ):
                    raise InvalidSpatialMetaError(("The key %s appears under "
                                                   "multiple "
                                                   "classifications"
                                                  ) % str(key_tup)
                                                 )
        
    def get_values_and_class(self, keys):
        '''Get all values and the classification for the provided key. 

        Parameters
        ----------
        keys : tuple
            The meta data key specifying a leaf element.
            
        Returns
        -------
        values  
            The value(s) or values for the given key. The number of values 
            returned depends on the classification (see 'get_n_vals').
             
        classification : str
            The classification of the meta data element
        '''
        for classification in self.get_classes():
            curr_dict = self._data[classification]
            for sub_key in keys:
                if not sub_key in curr_dict:
                    break
                curr_dict = curr_dict[sub_key]
            else:
                if isinstance(curr_dict, Mapping):
                    raise ValueError("The key does not specify a leaf element")
                return (curr_dict, classification)
        return (None, None)
        
    def _get_value(self, values, varying_dims, vox_idx):
        '''Get the single value for a given voxel index. '''
        shape = self.shape
        n_varying = len(varying_dims)
        if n_varying == 0:
            return values
            
        val_idx = 0
        for varying_idx in range(n_varying - 1, -1, -1):
            dim_idx = varying_dims[varying_idx]
            step_size = 1
            for inner_dim_idx in varying_dims[:varying_idx]:
                step_size *= shape[inner_dim_idx]
            val_idx += step_size * vox_idx[dim_idx]

        return values[val_idx]
        
    def get_value(self, keys, vox_idx):
        '''Get a single value for the provided meta data key and voxel index.
        
        Parameters
        ----------
        key : tuple
            The meta data key.
            
        vox_idx : tuple
            The voxel index associated with the meta data value        
        '''
        shape = self.shape
        if len(vox_idx) != len(shape):
            raise IndexError("Voxel index doesn't match the shape")
        if any(not 0 <= idx < shape[dim_idx] 
               for dim_idx, idx in enumerate(vox_idx)
              ):
            raise IndexError("Voxel index is out of range")
            
        values, classification = self.get_values_and_class(keys)
        varying_dims = self.get_varying_dims(classification)
        
        return self._get_value(values, varying_dims, vox_idx)
    
    def _get_expanded(self, values, varying_dims):
        '''Expand the values so there is one value per slice (if there is a 
        varying spatial dim) or per volume (if all varying dims are 
        non-spatial)'''
        #Build list of iterators to produce the indices we are interested in
        idx_iters = []
        for dim, dim_size in enumerate(self.shape):
            #Don't iterate over spatial dims that are non-varying
            if dim < 3 and not dim in varying_dims:
                idx_iters.append(range(1))
            else:
                idx_iters.append(range(dim_size))
        
        #Build list of values at each of these indices
        result = []
        for vox_idx in product(*idx_iters[::-1]):
            vox_idx = vox_idx[::-1]
            result.append(self._get_value(values, 
                                          varying_dims, 
                                          vox_idx)
                         )
        
        return result
    
    def _get_reduced(self, full_values, curr_varying, new_varying):
        '''Try to reduce the set of per slice/volume values generated from 
        the curr_varying dims using the new_varying dims. Returns None if the 
        reduction is not possible.'''
        shape = self.shape
        result = full_values
        #Iterate over the current varying dims in reverse order
        for curr_dim_idx in range(len(curr_varying) - 1, -1, -1):
            curr_dim = curr_varying[curr_dim_idx]
            #If the current varying dim is still varying in the new dims 
            #there is nothing to do
            if curr_dim in new_varying:
                continue
            
            #Otherwise we see if the values from lower dims are repeating
            period = 1
            for dim in curr_varying[:curr_dim_idx]:
                period *= shape[dim]
            if not period == 1:
                if is_repeating(result, period):
                    result = result[:period]
                else:
                    return None
            else:
                period = shape[curr_dim]
                if is_constant(result, period):
                    result = [x for x in result[::period]]
                else:
                    return None
            
        return result
    
    def simplify(self, keys):
        '''Try to reduce the number of varying dimensions of a single meta 
        data element by changing its classification. Return True if the 
        classification is changed, otherwise False. Looks for values that are 
        constant or repeating over some period. Constant elements with a 
        value of None will be deleted.
        '''
        values, curr_class = self.get_values_and_class(keys)
        if curr_class is None:
            raise KeyError("Keys not found: %s" % (keys,))
        curr_varying = self.get_varying_dims(curr_class)
        curr_n_varying = len(curr_varying)
        
        if curr_n_varying == 0:
            #If the element is already constant just check if the value is 
            #None
            if not values is None:
                return False
        elif is_constant(values):
            #Test if the values are globally constant
            val = values[0]
            if not val is None:
                self.set_nested('varies_over()', keys, values[0])
        elif curr_n_varying == 1:
            #If there is only one varying dim and we can't reclassify as 
            #globally constant then we are done
            return False
        else:
            #Build list of classes we could potentially reclassify to
            valid_classes = self.get_classes()
            potential_classes = []
            for n_varying in range(1, curr_n_varying):
                for varying_dims in combinations(curr_varying, n_varying):
                    cls = ('varies_over(' + 
                           ','.join(str(x) for x in varying_dims) + 
                           ')'
                          )
                    if cls in valid_classes:
                        potential_classes.append(cls)
                        
            #Sort the list of classes by the associated number of values
            potential_classes.sort(key=self.get_n_vals)
            
            #Expand the current set of values
            full_vals = self._get_expanded(values, curr_varying)
            
            #Try each of the potential reclassifications
            for new_cls in potential_classes:
                new_varying = self.get_varying_dims(new_cls)
                reduced_vals = self._get_reduced(full_vals, 
                                                 curr_varying,
                                                 new_varying)
                if not reduced_vals is None:
                    self.set_nested(new_cls, keys, reduced_vals)
                    break
            else:
                return False
            
        #We haven't returned False yet, so the element was reclassified
        self.remove_nested(curr_class, keys)
        return True
        
    def get_subset(self, dim, idx):
        '''Get a SpatialMeta object containing a subset of the meta data 
        corresponding to a single index along a given dimension.
        
        Parameters
        ----------
        dim : int
            The dimension we are taking the subset along.
            
        idx : int
            The position on the dimension `dim` for the subset.
        
        Returns
        -------
        result : SpatialMeta
            A new SpatialMeta object corresponding to the subset.
            
        '''
        
    @classmethod
    def from_sequence(klass, seq, dim, affine=None):
        '''Create an SpatialMeta object from a sequence of SpatialMeta 
        objects by joining them along the given dimension.
        
        Parameters
        ----------
        seq : sequence
            The sequence of SpatialMeta objects.
        
        dim : int
            The dimension to merge the along.
        
        affine : array
            The affine to use in the resulting SpatialMeta object. If None, 
            the affine from the first element in `seq` will be used.
            
        Returns
        -------
        result : SpatialMeta
            The result of merging the SpatialMeta objects in `seq` along the 
            dimension `dim`.
        '''
    
