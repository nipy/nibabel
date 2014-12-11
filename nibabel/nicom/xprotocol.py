# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Parse the "XProtocol" (or "EVAProtocol") meta data format found in a variety
of Siemens MR files.
"""

import re, string
from collections import MutableMapping, deque
from copy import deepcopy
import warnings

from ..externals import OrderedDict
from ..externals.six import iteritems


class InvalidElemError(Exception):
    '''Raised when trying to add an object without a `value` attribute to an
    `ElemDict`.'''


class ElemDict(MutableMapping):
    '''Ordered dict-like where each value is an "element", which is defined as
    any object which has a `value` attribute.

    When looking up an item in the dict, it is this `value` attribute that
    is returned. To get the element itself use the `get_elem` method.
    '''

    def __init__(self, mapping=None):
        self._elems = OrderedDict()
        if mapping is not None:
            for key, val in mapping.iteritems():
                if isinstance(val, dict):
                    val = ElemDict(val)
                self.set_elem(key, val)

    def __getitem__(self, key):
        return self._elems[key].value

    def __setitem__(self, key, val):
        if not hasattr(val, 'value'):
            raise InvalidElemError()
        self._elems[key] = val

    def __delitem__(self, key):
        del self._elems[key]

    def __iter__(self):
        return iter(self._elems)

    def __len__(self):
        return len(self._elems)

    def get_elem(self, key):
        return self._elems[key]


def iter_with_override(input_iter):
    '''Generator that wraps an iterable and just yields its values,
    unless an "override" iterable is sent to the generator. If
    provided, values from the override iterator are produced until it
    is exhausted at which point the original iterator is resumed. While
    multiple overlapping override iterators are allowed, it is not valid to
    send two override iterators back-to-back without consuming any elements
    in between.

    Can be used to "rewind" an iterator, however it is up to the
    consumer to cache the consumed elements and send them back.
    '''
    stack = deque([iter(input_iter)])
    while len(stack) > 0:
        try:
            elem = next(stack[0])
        except StopIteration:
            stack.popleft()
            continue

        # By default we just produce elements from the iterable
        override = yield elem

        # If the user sends back an iterable
        if override:
            yield None # This is returned to the send call
            stack.appendleft(iter(override))


class MissingOpenBraceError(Exception):
    '''Exception indicating an opening brace was not found.
    '''
    def __init__(self, message, popped_char):
        self.message = message
        self.popped_char = popped_char

    def __str__(self):
        return '%s : got %s' % (self.message, self.popped_char)


class MissingCloseBraceError(Exception):
    '''Exception indicating an opening brace was found but a closing
    brace was not.'''


class EmptyIterator(Exception):
    '''Exception indicating that the provided iterator was empty'''


def gen_brace_content(char_iter, brace_chars=('{', '}')):
    '''Generate characters for the text contained in the next set of
    braces. Strips the outermost braces in the process.

    The generator must be primed by calling `next` (or `send`) once before
    it is iterated over. This initial priming will return a None value if
    everything is okay or else raise an exception. An EmptyIterator will be
    raised if the char_iter is exhausted without seeing non-whitespace, and
    a MissingOpenBraceError will be raised if the opening brace is not found.

    A MissingClosingBrace error will be raised if the iterator is exhausted
    before the closing brace is found.

    Allows a string of characters that were just produced to be sent back in
    the same manner as `iter_with_override`, thus rewinding the iterator.

    Parameters
    ----------

    char_iter : generator
        A generator that produces characters and allows "overriding"
        (see `iter_with_override`)

    brace_chars : tuple
        The starting/ending brace characters.
    '''
    n_open = 0
    n_close = 0
    n_chars = 0
    leading_whitespace = []
    for char in char_iter:
        # Skip leading whitespace
        if char in string.whitespace and n_chars == 0:
            leading_whitespace.append(char)
            continue

        # Make sure we have initial brace and strip it. If not send back
        # any leading whitespace plus this char
        n_chars += 1
        if n_chars == 1:
            if char != brace_chars[0]:
                char_iter.send(leading_whitespace + [char])
                raise MissingOpenBraceError("Input is missing intitial open "
                                            "brace", char)
            n_open += 1

            # We are being "primed" so yield None
            yield None

            continue

        # Update paren counters
        if char == brace_chars[0]:
            n_open += 1
        elif char == brace_chars[1]:
            n_close += 1
            if n_close == n_open:
                break

        # Allow an override iterator to be sent in
        override = yield char
        if override:
            yield None
            # Adjust counts for brace chars that were sent back
            n_open -= override.count(brace_chars[0])
            n_close -= override.count(brace_chars[1])
            char_iter.send(override)

    else:
        if n_chars != 0:
            raise MissingCloseBraceError("Closing paren not found: %s" %
                                         ''.join(result))
        else:
            raise EmptyIterator()


class XProtocolParseError(Exception):
    '''Exception indicating an error occured while parsing the XProtocol'''


param_regex = '([A-Za-z]+)\."{1,2}([^"]*)"{1,2}'
'''Regex to capture the type and name of the parameter.'''


def parse_bool(char_iter, meta):
    val_str = ''.join(char_iter).strip()
    val_str = val_str.strip('"')
    val_str = val_str.lower()
    if val_str == '':
        return None
    if val_str == 'true':
        return True
    elif val_str == 'false':
        return False
    else:
        raise ValueError("Unrecognized value for bool parameter: %s" %
                         val_str)


def parse_string(char_iter, meta):
    val_str = ''.join(char_iter).strip()
    return val_str.strip('"')


def parse_quoted_str_list(char_iter, meta):
    val_str = ''.join(char_iter).strip()
    val_str.replace('""', '"')
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


def parse_map_element(char_iter, default_map):
    # Start with the default map
    default_keys = default_map.keys()
    result = deepcopy(default_map)

    # If values are provided, each will be contained in curly braces
    key_idx = 0
    while True:
        try:
            content = gen_brace_content(char_iter, ('{', '}'))
            next(content)
        except (EmptyIterator, MissingOpenBraceError):
            break
        key = default_keys[key_idx]
        key_idx += 1
        default_type = default_map[key].param_type
        # If the type is a mapping itself we are parsing a Map and
        # similarly if the type is a tuple we are parsing an Array
        if default_type == 'ParamMap':
            result[key] = parse_map_element(content, result[key])
        elif default_type == 'ParamArray':
            result[key] = parse_array_element(content, result[key][0])
        else:
            result[key] = type_parse_map[default_type](content, {})
    return result


def parse_array_element(char_iter, default_elem):
    result = []
    elem_idx = 0
    while True:
        try:
            content_iter = gen_brace_content(char_iter, ('{', '}'))
            next(content_iter)
            content = ''.join(content_iter) #TODO: Avoid this?
        except (EmptyIterator, MissingOpenBraceError):
            break

        if content == '':
            result.append(default_elem)
        elif default_elem.param_type == 'ParamMap':
            result.append(parse_map_element(iter_with_override(content),
                                            default_elem)
                         )
        elif default_elem.param_type == 'ParamArray':
            result.append(parse_array_element(iter_with_override(content),
                                              default_elem[0])
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
'''Map type string to the function that should parse the value when the
element is part of a nested type.'''


nested_types = set(['ParamArray',
                    'ParamMap',
                    'Pipe',
                    'PipeService',
                    'ParamFunctor',
                   ])
'''The set of types which can nest parameters'''


def parse_array(char_iter, meta):
    vals = []

    # Figure out the type of the elements in the array
    elem_type = meta['Default'][1].param_type

    # Elements that are nested types cannot be parsed as usual, so
    # we use specialized functions that have access to the "Default"
    # meta value and types
    if elem_type in nested_types:
        default_elem = meta['Default'][1]
        parse_args = (default_elem,)
    else:
        parse_args = ({},)
    while True:
        try:
            content_iter = gen_brace_content(char_iter, ('{', '}'))
            next(content_iter)
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
    return vals


def parse_map(char_iter, meta):
    result = ElemDict()
    for name, sub_val in gen_params(char_iter):
        result[name] = sub_val
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
'''Map types to the functions that parse their value.'''


class XProtocolElement(object):
    '''Represents a single leaf element in a XProtocol hierarchy.'''
    def __init__(self, param_type, value, meta):
        self.param_type = param_type
        self.value = value
        self.meta = meta

    def __repr__(self):
        return ('XProtocolElement(%r, %r, %r)' %
                (self.param_type, self.value, self.meta))


def get_elem(param_type, char_iter):
    '''Return the XProtocol element of the given type and content'''
    meta = {}
    parse_func = type_parse_map[param_type]

    # Loop looking for "meta values". These describe the actual
    # parameter value in some way. The identifier is enclosed in angle
    # brackets, but does not match the "ParamTYPE.NAME" format used for
    # normal parameters.
    while True:
        # Try to match angle brackets (possibly) denoting meta values,
        # otherwise we know the meta values have been exhausted and so
        # we break
        try:
            ident_iter = gen_brace_content(char_iter, ('<', '>'))
            next(ident_iter)
            ident = ''.join(ident_iter)
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
            meta_val_iter = gen_brace_content(char_iter, ('{', '}'))
            next(meta_val_iter)
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
        # elements).
        try:
            meta_val_ident_iter = gen_brace_content(meta_val_iter, ('<', '>'))
            next(meta_val_ident_iter)
            meta_val_ident = ''.join(meta_val_ident_iter)
            re_match = re.match(param_regex, meta_val_ident)
            meta_param_type, name = re_match.groups()

            # This assumes the first curly brace is on the next line
            meta_param_val_iter = gen_brace_content(char_iter,
                                                    ('{', '}')
                                                   )
            next(meta_param_val_iter)
            meta_val = (name, # TODO: Is this always an empty string?
                        get_elem(meta_param_type,
                                 meta_param_val_iter
                                )
                       )
        except (MissingOpenBraceError, EmptyIterator):
            # Otherwise we just keep the meta value as a string
            meta_val = ''.join(meta_val_iter)

        # Store the meta key/value
        meta[ident] = meta_val

    return XProtocolElement(param_type, parse_func(char_iter, meta), meta)


def gen_params(char_iter):
    while True:
        # Get identifier for next parameter, stop if char_iter is empty
        try:
            ident_iter = gen_brace_content(char_iter, ('<', '>'))
            next(ident_iter)
        except EmptyIterator:
            break
        except MissingOpenBraceError:
            break
        ident = ''.join(ident_iter)

        # Parse the param type and name, need to special case "XProtocol" root.
        # In older data sets the root name is "EVAProtocol", which we rename
        # to "XProtocol" for the sake of consistency.
        if ident in ('XProtocol', 'EVAProtocol'):
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
            next(val_iter)
        except EmptyIterator:
            raise XProtocolParseError("Unable to find value for %s" % name)

        # Parse the value text
        if param_type in type_parse_map:
            yield (name, get_elem(param_type, val_iter))
        else:
            # Issue a warning for unknown types
            warnings.warn("Unable to parse type '%s'" % param_type)
            # Consume the text for the value from the char_iter
            ''.join(val_iter)


def read_protos(input_str):
    '''Read zero or more XProtocol defintions from the given string (or
    iterator).

    Parameters
    ----------
    input_str : str or iter
        The string containing the xprotocol information.

    Returns
    -------
    protos : list
        A list of zero or more nested dict representations of the protocol
        data.

    remainder : iter
        A char iterator that produces the remainder of the input after reading
        any protocols.
    '''
    char_iter = iter_with_override(input_str)
    result = []

    for name, elem in gen_params(char_iter):
        if name != 'XProtocol':
            raise XProtocolParseError("Missing 'XProtocol' root element")
        elem_dict = ElemDict()
        elem_dict[name] = elem
        result.append(elem_dict)
    return result, char_iter
