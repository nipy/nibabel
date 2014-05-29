# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Parse the "XProtocol" meta data format found in a variety of Siemens MR files.
"""

import re, string
from collections import Mapping
from copy import deepcopy
import warnings

from ..externals import OrderedDict


def iter_with_override(input_iter):
    '''Generator that wraps an iterable and just yields its values,
    unless an "override" iterable is sent to the generator. If
    provided, values from the override iterator are produced until it
    is exhausted at which point the original iterator is resumed.

    Can be used to "rewind" an iterator, however it is up to the
    consumer to cache the consumed elements and send them back.
    '''
    for elem in input_iter:
        # By default we just produce elements from the iterable
        override = yield elem

        # If the user sends back an iterable
        if override:
            yield None # This is returned to the send call

            # Recursion to handle multiple overlapping overrides
            override_iter = iter_with_override(override)
            for override_elem in override_iter:
                sub_override = yield override_elem
                if sub_override:
                    yield None
                    override_iter.send(sub_override)


class MissingOpenBraceError(Exception):
    '''Exception indicating an opening brace was not found.
    '''
    def __init__(self, message, popped_char):
        self.message = message
        self.popped_char = popped_char


class MissingCloseBraceError(Exception):
    '''Exception indicating an opening brace was found but a closing
    brace was not.
    '''

class EmptyIterator(Exception):
    '''Exception indicating that the provided iterator was empty'''


def gen_brace_content(char_iter, brace_chars=('{', '}')):
    '''Generate characters for the text contained in the next set of
    braces. Strips the outermost braces in the process. Raises an
    EmptyIterator if the char_iter is exhausted without seeing
    non-whitespace. If the opening brace is not found the char_iter
    will be rewound to where it started.

    Allows an iterator to be sent back in the same manner as
    `iter_with_override`.
    Parameters
    ----------

    char_iter : generator
        A generator that produces characters and allows "overriding"
        (see `iter_with_override`)
    '''
    n_open = 0
    n_close = 0
    n_chars = 0
    leading_whitespace = []
    result = []
    for char in char_iter:
        # Skip leading whitespace
        if char in string.whitespace and n_chars == 0:
            leading_whitespace.append(char)
            continue

        n_chars += 1

        # Make sure we have initial brace and strip it. If not send back
        # any leading whitespace plus this char
        if n_chars == 1:
            if char != brace_chars[0]:
                char_iter.send(leading_whitespace + [char])
                raise MissingOpenBraceError("Input is missing intitial open "
                                            "brace", char)
            n_open += 1
            continue

        # Update paren counters
        if char == brace_chars[0]:
            n_open += 1
        elif char == brace_chars[1]:
            n_close += 1
            if n_close == n_open:
                break

        override = yield char
        if override:
            yield None # This it returned to the send call
            override_iter = iter_with_override(override)

            for override_elem in override_iter:
                sub_override = yield override_elem
                if sub_override:
                    yield None
                    override_iter.send(sub_override)

    else:
        if n_chars != 0:
            raise MissingCloseBraceError("Closing paren not found: %s" %
                                         ''.join(result))
        else:
            raise EmptyIterator()


class XProtocolParseError(Exception):
    '''Exception indicating an error occured while parsing the XProtocol
    '''


param_regex = '([A-Za-z]+)\."([^"]*)"'
'''Regex to capture the type and name of the param'''


def parse_bool(char_iter, meta):
    val_str = ''.join(char_iter).strip()
    val_str = val_str.lower()
    if val_str == '':
        return None
    if val_str == '"true"':
        return True
    elif val_str == '"false"':
        return False
    else:
        raise ValueError("Unrecognized value for bool parameter: %s" %
                         val_str)

def parse_string(char_iter, meta):
    val_str = ''.join(char_iter).strip()
    return val_str[1:-1]


def parse_quoted_str_list(char_iter, meta):
    val_str = ''.join(char_iter).strip()
    str_list = [x.strip() for x in val_str.split('"')]
    str_list = [x for x in str_list if not x == '']
    return str_list


def parse_long(char_iter, meta):
    val_str = ''.join(char_iter).strip()
    result = [int(x) for x in val_str.split()]
    if len(result) == 0:
        return None
    elif len(result) == 1:
        return result[0]
    else:
        return result


def parse_double(char_iter, meta):
    val_str = ''.join(char_iter).strip()
    result = [float(x) for x in val_str.split()]
    if len(result) == 0:
        return None
    elif len(result) == 1:
        return result[0]
    else:
        return result


def parse_map_element(char_iter, default_map, default_types):
    # Start with the default map
    default_keys = default_map.keys()
    result = deepcopy(default_map)

    # If values are provided, each will be contained in curly braces
    key_idx = 0
    while True:
        try:
            content = ''.join(gen_brace_content(char_iter, ('{', '}')))
        except (EmptyIterator, MissingOpenBraceError):
            break
        key = default_keys[key_idx]
        key_idx += 1
        default_type = default_types[key]
        # If the type is a mapping itself we are parsing a Map and
        # similarly if the type is a tuple we are parsing an Array
        if isinstance(default_type, Mapping):
            result[key] = parse_map_element(iter_with_override(content),
                                            result[key],
                                            default_type
                                           )
        elif isinstance(default_type, tuple):
            result[key] = parse_array_element(iter_with_override(content),
                                              result[key][0],
                                              default_type[0]
                                             )
        else:
            result[key] = type_parse_map[default_type](iter_with_override(content), {})
    return result


def parse_array_element(char_iter, default_elem, default_elem_type):
    result = []
    elem_idx = 0
    while True:
        try:
            content = ''.join(gen_brace_content(char_iter, ('{', '}')))
        except (EmptyIterator, MissingOpenBraceError):
            break

        if content == '':
            result.append(default_elem)
        elif isinstance(default_elem_type, Mapping):
            result.append(parse_map_element(iter_with_override(content),
                                            default_elem,
                                            default_elem_type)
                         )
        elif isinstance(default_elem_type, tuple):
            result.append(parse_array_element(iter_with_override(content),
                                              default_elem[0],
                                              default_elem_type[0])
                         )
        else:
            result.append(type_parse_map[default_type](iter(content), {}))

        elem_idx += 1

    return result


elem_type_parse_map = {'ParamBool' : parse_bool,
                       'ParamString' : parse_string,
                       'ParamChoice' : parse_string,
                       'ParamLong' : parse_long,
                       'ParamDouble' : parse_double,
                       'ParamMap' : parse_map_element,
                       'ParamArray' : parse_array_element,
                       'Dependency' : parse_quoted_str_list,
                       'Event' : parse_quoted_str_list,
                       'Method' : parse_quoted_str_list,
                       'Connection' : parse_quoted_str_list,
                      }


nested_types = set(['ParamArray',
                    'ParamMap',
                    'Pipe',
                    'PipeService',
                    'ParamFunctor',
                   ])
'''The set of types which can nest parameters'''


def parse_array(char_iter, meta, return_types=False):
    vals = []
    # Figure out the type of the elements in the array
    elem_type = meta['Default'][0]

    # Elements that are nested types cannot be parsed as usual, so
    # we use specialized functions that have access to the "Default"
    # meta value and types
    if elem_type in nested_types:
        default_val, default_types = meta['Default'][2]
        parse_args = (default_val, default_types)
    else:
        parse_args = ({},)
    while True:
        try:
            content_iter = \
                iter_with_override(''.join(gen_brace_content(char_iter,
                                                             ('{', '}')))
                                  )
            vals.append(elem_type_parse_map[elem_type](content_iter,
                                                       *parse_args)
                       )
        except EmptyIterator:
            # If no array elements have been found, generate one with
            # the empty iterator
            if len(vals) == 0:
                vals.append(elem_type_parse_map[elem_type](iter_with_override(''),
                                                           *parse_args)
                           )
            break
    if return_types:
        # For nested types we need to return the types of nested values
        if elem_type in nested_types:
            elem_type = default_types
        return vals, (elem_type,)
    else:
        return vals


def parse_map(char_iter, meta, return_types=False):
    result = OrderedDict()
    if return_types:
        result_types = OrderedDict()
    for name, sub_val, param_type in _gen_params(char_iter, True):
        result[name] = sub_val
        if return_types:
            result_types[name] = param_type
    if return_types:
        return (result, result_types)
    else:
        return result


type_parse_map = {'ParamBool' : parse_bool,
                  'ParamString' : parse_string,
                  'ParamChoice' : parse_string,
                  'ParamLong' : parse_long,
                  'ParamDouble' : parse_double,
                  'ParamArray' : parse_array,
                  'ParamMap' : parse_map,
                  'Pipe' : parse_map,
                  'PipeService' : parse_map,
                  'ParamFunctor' : parse_map,
                  'Dependency' : parse_quoted_str_list,
                  'Event' : parse_quoted_str_list,
                  'Method' : parse_quoted_str_list,
                  'Connection' : parse_quoted_str_list,
                 }
'''Map types to the functions that parse them'''


def get_val(param_type, char_iter, get_types=False):
    '''Return the value of the given type and content'''
    meta = {}
    parse_func = type_parse_map[param_type]

    # Loop looking for "meta values". These describe the actual
    # parameter value in some way.
    while True:
        # Try to match angle brackets (possibly) denoting meta values,
        # otherwise we know the meta values have been exhausted and so
        # we break
        try:
            ident = ''.join(gen_brace_content(char_iter, ('<', '>')))
        except (MissingOpenBraceError, EmptyIterator):
            break

        # Check if the 'ident' matches the pattern for parameter
        # identifiers in which case this must be a nested parameter
        # rather than a meta value. Thus we rewind the iterator and
        # break from our loop.
        re_match = re.match(param_regex, ident)
        if re_match:
            char_iter.send('<%s>' % ident)
            break

        # The meta values may or may not be enclosed in curly braces.
        try:
            meta_val_iter = \
                iter_with_override(''.join(gen_brace_content(char_iter,
                                                             ('{', '}')))
                                  )
        except MissingOpenBraceError:
            # If it is not in curly braces, it must start on the same
            # line as the identifier
            meta_val = []
            for char in char_iter:
                if char == '\n':
                    break
                meta_val.append(char)
            meta_val_iter = iter_with_override(''.join(meta_val).strip())

        # The meta val can be an "anonymous" parameter itself, in which
        # case we do need to parse it (i.e. with the 'Default' meta value
        # for Array types, it tells us how to parse the individual
        # elements)
        try:
            meta_val_ident = ''.join(gen_brace_content(meta_val_iter,
                                                       ('<', '>')
                                                      )
                                    )
            re_match = re.match(param_regex, meta_val_ident)
            meta_param_type, name = re_match.groups()

            # This assumes the first curly brace is on the next line
            meta_param_val_iter = gen_brace_content(char_iter,
                                                    ('{', '}')
                                                   )

            # If the meta value is a nested type get the nested types too
            if meta_param_type in nested_types:
                meta_val = (meta_param_type,
                            name,
                            get_val(meta_param_type,
                                    meta_param_val_iter,
                                    True
                                   )
                           )
            else:
                meta_val = (meta_param_type,
                            name,
                            get_val(meta_param_type,
                                    meta_param_val_iter
                                   )
                           )
        except (MissingOpenBraceError, EmptyIterator):
            # Otherwise we just keep the meta value as a string
            meta_val = ''.join(meta_val_iter)

        # Store the meta key/value
        meta[ident] = meta_val

    if param_type in nested_types:
        return parse_func(char_iter, meta, get_types)
    return parse_func(char_iter, meta)


def _gen_params(char_iter, return_types=False):
    while True:
        # Get identifier for next parameter, stop if char_iter is empty
        try:
            ident = ''.join(gen_brace_content(char_iter, ('<', '>')))
        except (EmptyIterator, MissingOpenBraceError):
            break

        # Parse the param type and name, need to special case
        # "XProtocol" tag
        if ident == 'XProtocol':
            param_type = 'ParamMap'
            name = 'XProtocol'
        else:
            re_match = re.match(param_regex, ident)
            if not re_match:
                raise XProtocolParseError("Error parsing element tag: %s" %
                                          ident)
            param_type, name = re_match.groups()

        # Get a char iterator for the text of the parameter value
        try:
            val_iter = gen_brace_content(char_iter, ('{', '}'))
        except EmptyIterator:
            raise XProtocolParseError("Unable to find value for %s" % name)

        # Parse the value text
        if param_type in type_parse_map:
            if return_types:
                # If the type is nested, we actually return a
                # representation that includes all of the nested types
                if param_type in nested_types:
                    val, param_type = get_val(param_type, val_iter, True)
                else:
                    val = get_val(param_type, val_iter)
                yield (name, val, param_type)
            else:
                yield (name, get_val(param_type, val_iter))
        else:
            warnings.warn("Unable to parse type '%s'" % param_type)
            ''.join(val_iter)


def generate_params(char_iter, return_types=False):
    '''Generate params from an XProtocol char iterator. By default
    yields the name and value for each parameter. If `return_types` is
    specified the type of each parameter will also be returned.'''
    char_iter = iter_with_override(char_iter)
    for result in _gen_params(char_iter, return_types):
        yield result
