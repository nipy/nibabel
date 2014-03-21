# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
DICOM anonymization tools.
"""

import sys, os, re, argparse, warnings, struct, uuid, hashlib
from os import path
from random import random
from copy import deepcopy

from ..py3k import asbytes
from . import csareader
from .structreader import Unpacker
from .utils import make_uid, as_to_years

try:
    import dicom
    from dicom.datadict import keyword_for_tag
except ImportError:
    pass


IMPLEMENTATION_CLASS_UID = '2.25.3434918901144735809493181392'
""" The UID used for the ImplementationClassUID tag."""


strict_exclude = ['Patient',
                  'AccessionNumber',
                  'Diagnoses',
                  'Physician',
                  'Operator',
                  'Date',
                  'Birth',
                  'Address',
                  'Institution',
                  'Station',
                  'SiteName',
                  'Comment',
                  'Phone',
                  'Telephone',
                  'Insurance',
                  'Religious',
                  'Language',
                  'History',
                  'Military',
                  'MedicalRecord',
                  'Ethnic',
                  'Occupation',
                  'UID',
                  'StudyDescription',
                  'SeriesDescription',
                  'ProtocolName',
                  'StudyID',
                  'RequestAttributesSequence',
                  'ContentSequence',
                  'DeviceSerialNumber',
                  'RequestedProcedureDescription',
                  'PerformedProcedureStepDescription',
                  'PerformedProcedureStepID',
                  'DerivationDescription',
                  'Private'
                 ]
""" Set of regular expressions used to determine which elements need to be
filtered in a strict setting."""


strict_force_include = ['SOPClassUID',
                        'ImageOrientationPatient',
                        'ImagePositionPatient',
                       ]
""" Set of regular expressions used to determine which elements to force
include in a strict setting."""


nonstrict_force_include = ['PatientSex',
                           'SeriesDescription',
                           'ProtocolName',
                          ] + strict_force_include
""" Set of regular expressions used to determine which elements to force
include in a non-strict setting. Includes elements that people often want
to keep but could be PHI."""


def make_keyword_regex_filter(exclude_res, force_include_res=None):
    """ Make a meta data filter using regular expressions.

    Parameters
    ----------
    exclude_res : sequence
        Sequence of regular expression strings. Any meta data where the key
        matches one of these expressions will be excluded, unless it matches
        one of the `force_include_res`.
    force_include_res : sequence
        Sequence of regular expression strings. Any meta data where the key
        matches one of these expressions will be included.

    Returns
    -------
    A callable which can be passed to DicomAnonymizer as the `filter_func`.
    """
    exclude_re = re.compile('|'.join(['(?:' + regex + ')'
                                      for regex in exclude_res])
                           )
    include_re = None
    if force_include_res:
        include_re = re.compile('|'.join(['(?:' + regex + ')'
                                          for regex in force_include_res])
                               )

    def keyword_regex_filter(elem):
        keyword = keyword_for_tag(elem.tag)
        return (exclude_re.search(keyword) and
                not (not include_re is None and include_re.search(keyword)))

    return keyword_regex_filter


default_filter_func = make_keyword_regex_filter(strict_exclude,
                                                strict_force_include)
""" Default filter function for DicomAnonymizer objects. Created by passing
the `strict_filtered` list to `make_key_regex_filter`."""


def remap_uid(elem):
    """ Remap the given UID to a new UID in a consistent (the same input
    results in the same output) but secure (it is not feasible to recreate
    the input from the output) manner."""
    if not elem.VR == 'UI':
        raise ValueError("Provided element is not a UID.")
    if elem.VM > 1:
        return [make_uid(entropy_srcs=[x]) for x in elem.value]
    else:
        return make_uid(entropy_srcs=[elem.value])


def non_strict_deident_age(age_str, min_age=1, max_age=89):
    """ Performs a non-strict deidentification of an DICOM age string. Gets
    rid of any information more specific that a year and clamps the value
    between `min_age` and `max_age`."""
    age = int(round(as_to_years(age_str)))
    if age <= min_age:
        return str(min_age)
    elif age >= max_age:
        return str(max_age)
    else:
        return str(age)


def non_strict_age_map(elem):
    """ If the age is greater than 89 it will be clamped to 89. This is often
    a sufficient level of anonymization. An example where this is not
    sufficient would be if the person has a condition where survival until
    their age is rare."""
    if not elem.VR == 'AS':
        raise ValueError("Element VR is not AS.")
    if elem.VM > 1:
        return [non_strict_deident_age(x) for x in elem.value]
    else:
        return non_strict_deident_age(elem.value)


def make_return_const(const_val):
    def return_const(elem):
        if elem.VM > 1:
            return [const_val] * elem.VM
        else:
            return const_val
    return return_const


def return_orig(elem):
    return elem.value


default_vr_replace_map = {'AE' : make_return_const('anon'),
                          'AS' : make_return_const('1'),
                          'AT' : return_orig, # Can't be PHI
                          'CS' : make_return_const('anon'),
                          'DA' : make_return_const('19000101'),
                          'DT' : make_return_const('19000101010101.000000'),
                          'LT' : make_return_const('anon'),
                          'PN' : make_return_const('anon'),
                          'SH' : make_return_const('anon'),
                          'TM' : make_return_const('010101.000000'),
                          'UI' : remap_uid,
                          'UT' : make_return_const('anon'),
                         }
""" Maps Value Representations (VRs) to callables that take the DICOM element
and return the anonymized replacement value."""


class StandardDicomMapping(object):
    """Callable object which takes standard DICOM elements and returns the
    tuple (is_filtered, filtered_value).

    Parameters
    ----------
    injection_map : dict
        Maps DICOM keywords to the value that element should be set to. Any
        element included here will not be filtered further.

    filter_func : callable
        Callable that takes DICOM element and returns True if it should be
        filtered. If None defaults to `default_filter_func`.

    elem_replace_map : dict
        Maps DICOM keywords to functions that provide the replacement values
        for filtered elements. Takes precedence over the `vr_replace_map`.

    vr_replace_map : dict
        Map DICOM Value Representations to replacement values for filtered
        elements. If None defaults to defualt_vr_replace_map.
    """

    def __init__(self, injection_map=None, filter_func=None,
                 elem_replace_map=None, vr_replace_map=None):
        if injection_map is None:
            self.injection_map = {}
        else:
            self.injection_map = injection_map
        if filter_func is None:
            self.filter_func = default_filter_func
        else:
            self.filter_func = filter_func
        if elem_replace_map is None:
            self.elem_replace_map = {}
        else:
            self.elem_replace_map = elem_replace_map
        if vr_replace_map is None:
            self.vr_replace_map = default_vr_replace_map
        else:
            self.vr_replace_map = vr_replace_map

    def __call__(self, elem):
        keyword = keyword_for_tag(elem.tag)
        if keyword in self.injection_map:
            return (True, self.injection_map[keyword])
        if self.filter_func(elem):
            if keyword in self.elem_replace_map:
                filtered_val = self.elem_replace_map[keyword](elem)
            elif elem.VR in self.vr_replace_map:
                filtered_val = self.vr_replace_map[elem.VR](elem)
            else:
                filtered_val = None
            return (True, filtered_val)
        else:
            return (False, None)


def csa_img_mosaic_mapper(elem):
    """ Specialized filter function for CSA Image header in mosaic images.

    Zeros out any unused portions of the CSA header as these can potentially
    contain PHI from a reused portion of memory.
    """
    try:
        csa = csareader.read(elem.value)
        if ('NumberOfImagesInMosaic' in csa['tags'] and
            len(csa['tags']['NumberOfImagesInMosaic']['items']) != 0
           ):
            return csareader.write(elem.value)
        else:
            return None
    except Exception:
        return None


non_strict_priv_elems = {'SIEMENS CSA HEADER' : \
                            {dicom.tag.Tag(0x29, 0x1010) : \
                                csa_img_mosaic_mapper,
                            }
                        }


class PrivateDicomMapping(object):
    """Callable object which takes a private element and returns the tuple
    (is_filtered, filtered_value).

    Parameters
    ----------
    priv_elem_map : dict
        Nested mapping of private creator values, to associated DICOM tags,
        to callables that return the filtered value. Any private element not
        included here (or not a private creator element) is considered
        unknown.

    unknown_filter : callable
        Callable that takes a private element not specified in
        `priv_elem_map` and returns True if it should be filtered. If not
        specified it will filter all unknown elements.

    vr_replace_map : dict
        Map DICOM Value Representations to replacement values for filtered
        unknown elements. If None defaults to defualt_vr_replace_map.
    """

    def __init__(self, priv_elem_map=None, unknown_filter=None,
                 vr_replace_map=None):
        if priv_elem_map is None:
            self.priv_elem_map = {}
        else:
            self.priv_elem_map = priv_elem_map
        self.unknown_filter = unknown_filter
        if vr_replace_map is None:
            self.vr_replace_map = default_vr_replace_map
        else:
            self.vr_replace_map = vr_replace_map
        self.resolved_priv_elem_map = None

    def prepare(self, dcm_data):
        '''Just clear any known private creators. Should be called before
        processing a new dataset.'''
        self.resolved_priv_elem_map = {}

    def __call__(self, elem):
        if elem.name == "Private Creator":
            # Use the private creator to resolve any associated tags in
            # priv_elem_map
            if elem.value in self.priv_elem_map:
                for orig_tag, func in self.priv_elem_map[elem.value].iteritems():
                    new_elem_num = ((orig_tag.elem & 0xff) |
                                    (elem.tag.elem * 16**2))
                    new_tag = dicom.tag.Tag(elem.tag.group, new_elem_num)
                    self.resolved_priv_elem_map[new_tag] = func

            # Keep all private creator elements
            return (False, None)

        # Check if the element is in the resolved_priv_elem_map
        if elem.tag in self.resolved_priv_elem_map:
            return (True, self.resolved_priv_elem_map[elem.tag](elem))
        elif not self.unknown_filter is None:
            if self.unknown_filter(elem):
                if elem.VR in self.vr_replace_map:
                    filtered_val = self.vr_replace_map[elem.VR](elem)
                else:
                    filtered_val = None
                return (True, filtered_val)
            else:
                return (False, None)
        else:
            return (True, None)


class DicomAnonymizer(object):
    """ Object is a callable that takes a single argument, the DICOM dataset
    to anonymize.

    Parameters
    ----------
    standard_mapping : callable
        Callable that takes a standard DICOM element and returns a tuple
        '(is_fitlered, filtered_value)'. If 'is_filtered' is True and
        'filtered_value' is None the element will be deleted. If the
        callable has a method 'prepare' it will be passed the full
        DICOM dataset prior to any calls.

    private_mapping : callable
        Callable that takes a private DICOM element and returns a tuple
        '(is_fitlered, filtered_value)'. If 'is_filtered' is True and
        'filtered_value' is None the element will be deleted. If the
        callable has a method 'prepare' it will be passed the full
        DICOM dataset prior to any calls.
    """

    def __init__(self, standard_mapping=None, private_mapping=None):
        if standard_mapping is None:
            self.standard_mapping = StandardDicomMapping()
        else:
            self.standard_mapping = standard_mapping
        if private_mapping is None:
            self.private_mapping = PrivateDicomMapping()
        else:
            self.private_mapping = private_mapping

    def __call__(self, dcm_data):
        '''Anonymize the given `dcm_data` in place.'''
        # Allow the mapping objects a chance to prepare for the dataset we
        # are about to process
        if hasattr(self.standard_mapping, 'prepare'):
            self.standard_mapping.prepare(dcm_data)
        if hasattr(self.private_mapping, 'prepare'):
            self.private_mapping.prepare(dcm_data)

        # Build list of elements that need filtering
        filtered_elems = []
        for elem in dcm_data:
            # Handle private elements
            if elem.tag.group % 2 == 1:
                is_filtered, filtered_val = self.private_mapping(elem)
                if is_filtered:
                    filtered_elems.append((elem, filtered_val))
            elif isinstance(elem.value, dicom.sequence.Sequence):
                # Save copies of our current mapping objects
                orig_std_mapping = deepcopy(self.standard_mapping)
                orig_priv_mapping = deepcopy(self.private_mapping)

                # Handle nested sequences with recursion
                for ds in elem.value:
                    self(ds)

                # Restore the mapping objects
                self.standard_mapping = orig_std_mapping
                self.private_mapping = orig_priv_mapping
            else:
                is_filtered, filtered_val = self.standard_mapping(elem)
                if is_filtered:
                    filtered_elems.append((elem, filtered_val))

        # Iterate over the filtered elements, updating the dataset in place
        for elem, value in filtered_elems:
            if value is None:
                del dcm_data[elem.tag]
            else:
                elem.value = value

        # Replace the 'file_meta' dataset if it exists
        if hasattr(dcm_data, 'file_meta'):
            file_meta = dicom.dataset.Dataset()
            file_meta.MediaStorageSOPClassUID = dcm_data.SOPClassUID
            file_meta.MediaStorageSOPInstanceUID = dcm_data.SOPInstanceUID
            file_meta.ImplementationClassUID = IMPLEMENTATION_CLASS_UID
            if dcm_data.is_little_endian and dcm_data.is_implicit_VR:
                file_meta.add_new((2, 0x10), 'UI', dicom.UID.ImplicitVRLittleEndian)
            elif dcm_data.is_little_endian and not dcm_data.is_implicit_VR:
                file_meta.add_new((2, 0x10), 'UI', dicom.UID.ExplicitVRLittleEndian)
            elif not dcm_data.is_little_endian and not dcm_data.is_implicit_VR:
                file_meta.add_new((2, 0x10), 'UI', dicom.UID.ExplicitVRBigEndian)
            else:
                raise NotImplementedError("pydicom has not been verified for "
                                          "Big Endian with Implicit VR")
            dcm_data.file_meta = file_meta

        # Get rid of any preamble if it exists
        if hasattr(dcm_data, 'preamble'):
            dcm_data.preamble = None
