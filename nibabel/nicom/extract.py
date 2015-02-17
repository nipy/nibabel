# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Defines MetaExtractor class which will pull meta data from a DICOM data set
into a mapping of key/value pairs that is ready to be serialized (e.g. into
JSON).
"""
from __future__ import division, print_function, absolute_import

import struct, warnings
from collections import defaultdict

from . import csareader, ascconv
from .utils import find_private_section
from .. import xpparse as xpp
from ..externals import OrderedDict
from ..externals.six import iteritems

try:
    import dicom
    from dicom.datadict import keyword_for_tag
    # This is needed to allow extraction on files with invalid values (e.g. too
    # long of a decimal string)
    dicom.config.enforce_valid_values = False
except ImportError:
    pass

try:
    import chardet
    have_chardet = True
except ImportError:
    pass

class PrivateTranslator(object):
    """ Object for translating a private element into a dictionary."""

    def __init__(self, name, group_no, elem_offset, creators, trans_func):
        self.name = name
        self.group_no = group_no
        self.elem_offset = elem_offset
        self.creators = creators
        self.trans_func = trans_func

    def translate(self, elem):
        return self.trans_func(elem)


def csa_image_trans_func(elem):
    '''Function for translating the CSA image sub header.'''
    return csareader.header_to_key_val_mapping(csareader.read(elem.value))


csa_image_trans = PrivateTranslator('CsaImage',
                                    0x29,
                                    0x10,
                                    ['SIEMENS CSA HEADER',
                                     'CSA Image Header Info'],
                                    csa_image_trans_func)
'''Translator for the CSA image sub header.'''


def csa_series_trans_func(elem):
    '''Function for parsing the CSA series sub header.'''
    csa_dict = csareader.header_to_key_val_mapping(csareader.read(elem.value))

    # Exactly where the XProtocol and ASCCONV data is stored depends on the
    # version of Syngo.
    if 'MrPhoenixProtocol' in csa_dict:
        # This is a "newer" data set, everything is in 'MrPhoenixProtocol'
        outer_xprotos = xpp.parse(csa_dict['MrPhoenixProtocol'])
        assert len(outer_xprotos) == 1
        xproto_dict = outer_xprotos[0]
        for param in xproto_dict['blocks'][0]['value']:
            if param['name'] == 'Protocol0':
                inner_str, ascconv_str = xpp.split_ascconv(param['value'])
                param['value'] = xpp.parse(xpp.strip_twin_quote(inner_str))
                break
        else:
            raise ValueError("Unable to find inner XProtocol and ASCCONV")
        ascconv_dict = ascconv.parse_ascconv(''.join(remainder),
                                             'MrPhoenixProtocol')
        del csa_dict['MrPhoenixProtocol']
    elif 'MrProtocol' in csa_dict and 'MrEvaProtcol' in csa_dict:
        # This is an "older" data set, XProtocol is in 'MrEvaProtcol' and
        # ASCCONV is in 'MrProtocol'
        xprotos = xpp.parse(csa_dict['MrEvaProtocol'])
        assert len(xprotos) == 1
        xproto_dict = xprotos[0]
        ascconv_dict = ascconv.parse_ascconv(csa_dict['MrProtocol'],
                                             'MrProtocol')
        del csa_dict['MrEvaProtocol']
        del csa_dict['MrProtocol']
    else:
        raise ValueError("Unrecognized structure for CSA Series Element")

    # We try to normalize these discrepacies by always putting the parsed
    # results into the same locations
    csa_dict['XProtocol'] = xproto_dict
    csa_dict['ASCCONV'] = ascconv_dict

    return csa_dict


csa_series_trans = PrivateTranslator('CsaSeries',
                                     0x29,
                                     0x20,
                                     ['SIEMENS CSA HEADER',
                                      'CSA Series Header Info'],
                                     csa_series_trans_func)
'''Translator for parsing the CSA series sub header.'''


default_translators = (csa_image_trans,
                       csa_series_trans,
                      )
'''Default translators for MetaExtractor.'''


def is_ascii(in_str):
    '''Return true if the given string is valid ASCII.'''
    if all(' ' <= c <= '~' for c in in_str):
        return True
    return False


def get_text(byte_str):
    '''If the given byte string contains text data return it as unicode,
    otherwise return None.

    If the 'chardet' package is installed, this will be used to detect the
    text encoding. Otherwise the input will only be decoded if it is ASCII.
    '''
    if have_chardet:
        match = chardet.detect(byte_str)
        if match['encoding'] is None:
            return None
        else:
            return byte_str.decode(match['encoding'])
    else:
        if not is_ascii(byte_str):
            return None
        else:
            return byte_str.decode('ascii')


def tag_to_str(tag):
    '''Convert a DICOM tag to a string representation using the group and
    element hex values seprated by an underscore.'''
    return '%#X_%#X' % (tag.group, tag.elem)


def make_vr_conv_filter(vr_to_conv_dict):
    def vr_conv_filter(dcm, elem, value):
        conv_func = vr_to_conv_dict.get(elem.VR)
        if conv_func is None:
            return value
        elif elem.VM > 1:
            return [conv_func(x) for x in value]
        else:
            return conv_func(value)
    return vr_conv_filter


default_conversions = {'DS' : float,
                       'IS' : int,
                       'AT' : str, # TODO: Make this result less ambiguous?
                       'OW' : get_text,
                       'OB' : get_text,
                       'OW or OB' : get_text,
                       'UN' : get_text,
                      }
'''Default dictionary mapping value representations to functions used to
convert the value'''


def make_tag_ignore_filter(tag_list):
    def tag_ignore_filter(dcm, elem, value):
        if elem.tag in tag_list:
            return None
        return value
    return tag_ignore_filter


def ignore_empty_str(dcm, elem, value):
    if value == '':
        return None
    return value

default_ignore_tags = [dicom.tag.Tag((0x7fe0, 0x10))]
'''Default set of dicom tags to ignore'''


default_filters = [ignore_empty_str,
                   make_tag_ignore_filter(default_ignore_tags),
                   make_vr_conv_filter(default_conversions),
                  ]
'''Default list of filter functions for the `MetaExtractor` class'''


unpack_vr_map = {'SL' : 'i',
                 'UL' : 'I',
                 'FL' : 'f',
                 'FD' : 'd',
                 'SS' : 'h',
                 'US' : 'H',
                 'US or SS' : 'H',
                 }
'''Dictionary mapping value representations to corresponding format strings for
the struct.unpack function.'''


class MetaExtractor(object):
    '''Callable object for extracting meta data from a dicom dataset.

    Parameters
    ----------
    translators : list
        A list of `Translator` objects each of which can convert a
        private DICOM element into a dictionary.

    filters : list
        A list of functions that can filter the values being extracted.
        The functions take three arguments: the dicom dataset, the dicom
        element, and the value (which may have already been modified by
        previous filters). Any filter returning `None` will cause that element
        to be ignored.

    warn_on_trans_except : bool
        Convert any exceptions from translators into warnings.

    Notes
    -----
    The `translators` and `filters` arguments will default to the module
    variables `default_translators` and `default_filters` respectively, if
    they are `None`. To disable these you must pass an empty list.
    '''

    def __init__(self, translators=None, filters=None,
                 warn_on_trans_except=True):
        self.translators = translators
        if self.translators is None:
            self.translators = default_translators
        self.filters = filters
        if filters is None:
            self.filters = default_filters
        self.warn_on_trans_except = warn_on_trans_except

    def _get_elem_key(self, elem):
        '''Get the key for any non-translated elements.'''
        # Use standard DICOM keywords if possible
        key = keyword_for_tag(elem.tag)

        # For private tags we take elem.name and convert to camel case
        if key == '':
            key = elem.name
            if key.startswith('[') and key.endswith(']'):
                key = key[1:-1]
            tokens = [token[0].upper() + token[1:]
                      for token in key.split()]
            key = ''.join(tokens)

        return key

    def _get_elem_value(self, dcm, elem):
        '''Get the value for any non-translated elements'''
        if elem.VR in unpack_vr_map and isinstance(elem.value, str):
            # TODO: I guess this is a workaround for a pydicom bug, need to
            # check if it is still an issue (and if it is, get it fixed
            # upstream)
            if elem.VM == 1:
                result = struct.unpack(unpack_vr_map[elem.VR], elem.value)[0]
            else:
                result = list(struct.unpack(unpack_vr_map[elem.VR], elem.value))
        else:
            if elem.VM == 1:
                result = elem.value
            else:
                result = elem.value[:]

        # Apply filters to the value
        for filter_func in self.filters:
            result = filter_func(dcm, elem, result)
            if result is None:
                break

        return result

    def __call__(self, dcm):
        '''Extract the meta data from a DICOM dataset.

        Parameters
        ----------
        dcm : dicom.dataset.Dataset
            The DICOM dataset to extract the meta data from.

        Returns
        -------
        meta : dict
            A dictionary of extracted meta data.

        Notes
        -----
        Non-private tags use the DICOM keywords as keys. Translators use their
        name.
        '''
        # Convert text types to unicode
        dcm.decode()

        # Make dict mapping tags to tranlators
        trans_map = {}
        for translator in self.translators:
            for creator in translator.creators:
                sect_start = find_private_section(dcm,
                                                  translator.group_no,
                                                  creator)
                if sect_start is not None:
                    break
            else:
                continue
            tag = dicom.tag.Tag(translator.group_no,
                                sect_start + translator.elem_offset)
            if tag in trans_map:
                raise ValueError('More than one translator given for tag: '
                                 '%s' % tag)
            trans_map[tag] = translator

        result = []
        for elem in dcm:
            # Get the name for non-translated elements
            name = self._get_elem_key(elem)

            # If there is a translator for this element, use it
            if elem.tag in trans_map:
                translator = trans_map[elem.tag]
                try:
                    meta = translator.translate(elem)
                except Exception as e:
                    if self.warn_on_trans_except:
                        warnings.warn("Exception from translator %s: %s" %
                                      (translator.name, str(e)))
                    else:
                        raise
                else:
                    if meta:
                        result.append((translator.name, meta, elem.tag))

            # Handle elements that are sequences with recursion
            elif isinstance(elem.value, dicom.sequence.Sequence):
                value = []
                for val in elem.value:
                    value.append(self(val))
                result.append((name, value, elem.tag))
            else:
                val = self._get_elem_value(dcm, elem)
                if val is not None:
                    result.append((name, val, elem.tag))

        # Handle name collisions (can only happen from non-translated private
        # tags)
        name_counts = defaultdict(int)
        for elem in result:
            name_counts[elem[0]] += 1
        result_dict = OrderedDict()
        for name, value, tag in result:
            if name_counts[name] > 1:
                name = name + '_' + tag_to_str(tag)
            result_dict[name] = value

        return result_dict
