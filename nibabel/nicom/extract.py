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
into a mapping of key/value pairs.
"""
from __future__ import division, print_function, absolute_import

import struct, warnings
from collections import namedtuple, defaultdict

from . import csareader, xprotocol, phoenix, anon
from ..externals import OrderedDict

try:
    import dicom
    from dicom.datadict import keyword_for_tag
    # This is needed to allow extraction on files with invalid values (e.g. too
    # long of a decimal string)
    dicom.config.enforce_valid_values = False
except ImportError:
    pass


class PrivateTranslator(object):
    """ Object for translating a private element into a dictionary."""

    def __init__(self, name, group_no, elem_offset, creator, trans_func,
                 elem_filter=None):
        self.name = name
        self.group_no = group_no
        self.elem_offset = elem_offset
        self.creator = creator
        if elem_filter is None:
            elem_filter = anon.default_filter_func
        self.elem_filter = elem_filter

    def translate(self, elem):
        raise NotImplementedError


def csa_image_trans_func(elem):
    '''Function for translating the CSA image sub header.'''
    return csareader.header_to_key_val_mapping(csareader.read(elem.value))


csa_image_trans = Translator('CsaImage',
                             0x29,
                             0x10,
                             'SIEMENS CSA HEADER',
                             csa_image_trans_func)
'''Translator for the CSA image sub header.'''


def csa_series_trans_func(elem):
    '''Function for parsing the CSA series sub header.'''
    csa_dict = csareader.header_to_key_val_mapping(csareader.read(elem.value))

    # If there is a phoenix protocol, parse it and add it under the csa_dict
    phx_src = None
    if 'MrPhoenixProtocol' in csa_dict:
        phx_src = 'MrPhoenixProtocol'
    elif 'MrProtocol' in csa_dict:
        phx_src = 'MrProtocol'

    if not phx_src is None:
        phoenix_dict = parse_phoenix_prot(phx_src, csa_dict[phx_src])
        del csa_dict[phx_src]
        csa_dict['MrPhoenixProtocol'] = phoenix_dict

    return csa_dict


csa_series_trans = Translator('CsaSeries',
                              0x29,
                              0x20,
                              'SIEMENS CSA HEADER',
                              csa_series_trans_func)
'''Translator for parsing the CSA series sub header.'''


default_translators = (csa_image_trans,
                       csa_series_trans,
                      )
'''Default translators for MetaExtractor.'''


def tag_to_str(tag):
    '''Convert a DICOM tag to a string representation using the group and
    element hex values seprated by an underscore.'''
    return '%#X_%#X' % (tag.group, tag.elem)


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


default_conversions = {'DS' : float,
                       'IS' : int,
                       'AT' : str,
                      }


def make_unicode(in_str):
    '''Try to convetrt in_str to unicode'''
    for encoding in ('utf8', 'latin1'):
        try:
            result = unicode(in_str, encoding=encoding)
        except UnicodeDecodeError:
            pass
        else:
            break
    else:
        raise ValueError("Unable to determine string encoding: %s" % in_str)
    return result


class MetaExtractor(object):
    '''Callable object for extracting meta data from a dicom dataset.
    Initialize with an anonymizer and a set of translators and type
    conversions.

    Parameters
    ----------
    anonymizer : callable
        A callable that takes a DICOM dataset and returns the anonymized
        version.

    translators : sequence
        A sequence of `Translator` objects each of which can convert a
        private DICOM element into a dictionary. Overrides the anonymizer.

    conversions : dict
        Mapping of DICOM value representation (VR) strings to callables
        that perform some conversion on the value

    warn_on_trans_except : bool
        Convert any exceptions from translators into warnings.
    '''

    def __init__(self, anonymizer=None, translators=None, conversions=None,
                 warn_on_trans_except=True):
        if ignore_rules is None:
            self.ignore_rules = default_ignore_rules
        else:
            self.ignore_rules = ignore_rules
        if translators is None:
            self.translators = default_translators
        else:
            self.translators = translators
        if conversions is None:
            self.conversions = default_conversions
        else:
            self.conversions = conversions
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

    def _get_elem_value(self, elem):
        '''Get the value for any non-translated elements'''
        if elem.VR in unpack_vr_map and isinstance(elem.value, str):
            if elem.VM == 1:
                return struct.unpack(unpack_vr_map[elem.VR], elem.value)[0]
            else:
                return list(struct.unpack(unpack_vr_map[elem.VR], elem.value))

        if elem.VR in self.conversions:
            if elem.VM == 1:
                return self.conversions[elem.VR](elem.value)
            else:
                return [self.conversions[elem.VR](val) for val in elem.value]

        if elem.VM == 1:
            return elem.value
        else:
            return elem.value[:]

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
        Non-private tags use the DICOM keywords as keys. Translators have their
        name, followed by a dot, prepended to the keys of any meta elements
        they produce. Values are unchanged, except when the value
        representation is 'DS' or 'IS' (decimal/integer strings) they are
        converted to float and int types.
        '''
        standard_meta = []
        trans_meta_dicts = OrderedDict()

        # Make dict mapping tags to tranlators
        trans_map = {}
        for translator in self.translators:
            if translator.tag in trans_map:
                raise ValueError('More than one translator given for tag: '
                                 '%s' % translator.tag)
            trans_map[translator.tag] = translator

        for elem in dcm:
            if isinstance(elem.value, str) and elem.value.strip() == '':
                continue

            # Get the name for non-translated elements
            name = self._get_elem_key(elem)

            # If it is a private creator element, handle any corresponding
            # translators
            if elem.name == "Private Creator":
                moves = []
                for curr_tag, translator in trans_map.iteritems():
                    if translator.priv_creator == elem.value:
                        new_elem = ((translator.tag.elem & 0xff) |
                                    (elem.tag.elem * 16**2))
                        new_tag = dicom.tag.Tag(elem.tag.group, new_elem)
                        if new_tag != curr_tag:
                            if (new_tag in trans_map or
                                any(new_tag == move[0] for move in moves)
                               ):
                                raise ValueError('More than one translator '
                                                 'for tag: %s' % new_tag)
                            moves.append((curr_tag, new_tag))
                for curr_tag, new_tag in moves:
                    trans_map[new_tag] = trans_map[curr_tag]
                    del trans_map[curr_tag]

            # If there is a translator for this element, use it
            if elem.tag in trans_map:
                try:
                    meta = trans_map[elem.tag].trans_func(elem)
                except Exception, e:
                    if self.warn_on_trans_except:
                        warnings.warn("Exception from translator %s: %s" %
                                      (trans_map[elem.tag].name,
                                       str(e)))
                    else:
                        raise
                else:
                    if meta:
                        trans_meta_dicts[trans_map[elem.tag].name] = meta
            # Otherwise see if we are supposed to ignore the element
            elif any(rule(elem) for rule in self.ignore_rules):
                continue
            # Handle elements that are sequences with recursion
            elif isinstance(elem.value, dicom.sequence.Sequence):
                value = []
                for val in elem.value:
                    value.append(self(val))
                standard_meta.append((name, value, elem.tag))
            # Otherwise just make sure the value is unpacked
            else:
                standard_meta.append((name,
                                      self._get_elem_value(elem),
                                      elem.tag
                                     )
                                    )

        # Handle name collisions
        name_counts = defaultdict(int)
        for elem in standard_meta:
            name_counts[elem[0]] += 1
        result = OrderedDict()
        for name, value, tag in standard_meta:
            if name_counts[name] > 1:
                name = name + '_' + tag_to_str(tag)
            result[name] = value

        # Inject translator results
        for trans_name, meta in trans_meta_dicts.iteritems():
            for name, value in meta.iteritems():
                name = '%s.%s' % (trans_name, name)
                result[name] = value

        # Make sure all string values are unicode
        for name, value in result.iteritems():
            if isinstance(value, str):
                result[name] = make_unicode(value)
            elif (isinstance(value, list) and
                  len(value) > 0 and
                  isinstance(value[0], str)):
                result[name] = [make_unicode(val) for val in value]

        return result
