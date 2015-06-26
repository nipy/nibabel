""" PLY parser for Siemens xprotocol XML-like format
"""
from __future__ import print_function, absolute_import

import re

from .externals.ply import lex
from .externals.ply import yacc


def find_column(input, lexpos):
    """ Get line column number given input string `input` and lex pos `lexpos`

    Parameters
    ----------
    input : str
        The input text string.
    lexpos : int
        The position in the character stream

    Returns
    -------
    column : int
        Index to the character in the input line
    """
    last_cr = input.rfind('\n', 0, lexpos)  # -1 if not found
    return lexpos - last_cr - 1


class XProtocolSymbols(object):
    # Tags that don't have a name, just a type
    anon_tags = set(['XProtocol',
                     'EVAProtocol',
                     'Control',
                     'Param',
                     'Pos',
                     'Repr',
                     'Line',
                     'EVAStringTable',
                    ])

    # Named tags have both a type and a name
    named_tags = set(['ParamBool',
                      'ParamLong',
                      'ParamDouble',
                      'ParamString',
                      'ParamArray',
                      'ParamMap',
                      'ParamChoice',
                      'ParamFunctor',
                      'ParamCardLayout',
                      'PipeService',
                      'EVACardLayout',
                      'Connection',
                      'Dependency',
                      'Event',
                      'Method',
                     ])

    # Use all uppercase version of tags as the token names in PLY
    tokens = [
        'TAG',
        'NAMED_TAG',
        'WHITESPACE',
        'INTEGER',
        'FLOAT',
        'MULTI_STRING',
        'TRUE',
        'FALSE',
    ] + [x.upper() for x in anon_tags] + [x.upper() for x in named_tags]

    literals = '{}'

    def __init__(self, error_mode='strict'):
        """ Build lexer and parser with given `error_mode`

        Parameters
        ----------
        error_mode : {'strict', 'forgiving'}
            'strict' gives SyntaxErrors for a lexing or parsing error.
            'forgiving' tries to skip past the errors.
        """
        if error_mode not in ('strict', 'forgiving'):
            raise ValueError('Error mode should be "strict" or "forgiving"')
        self.lexer = lex.lex(module=self)
        self.parser = yacc.yacc(debug=False, module=self)
        self.error_mode = error_mode

    # Anon tag
    def t_TAG(self, t):
        r'<(?P<tagtype>[A-Za-z_][\w_]*)>'
        tag_type = t.lexer.lexmatch.group('tagtype')
        if tag_type in self.anon_tags:
            token = tag_type.upper()
        else:
            token = 'TAG'
        t.type = token
        t.value = (tag_type, None)
        return t

    # Named tag
    def t_NAMED_TAG(self, t):
        r'<(?P<tagtype>[A-Za-z_][\w_]*)\."(?P<tagname>.*?)">'
        match = t.lexer.lexmatch
        tag_type = match.group('tagtype')
        if tag_type in self.named_tags:
            token = tag_type.upper()
        else:
            token = 'NAMED_TAG'
        t.type = token
        tag_name = match.group('tagname')
        t.value = (tag_type, tag_name)
        return t

    # Whitespace
    def t_WHITESPACE(self, t):
        r'\s+'
        t.lexer.lineno += t.value.count("\n")

    # Floating literal
    def t_FLOAT(self, t):
        r'[+-]?(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?'
        t.value = float(t.value)
        return t

    # Integer literal
    def t_INTEGER(self, t):
        r'[-]?[0-9]+'
        t.value = int(t.value)
        return t

    def t_TRUE(self, t):
        r'"true"'
        t.value = True
        return t

    def t_FALSE(self, t):
        r'"false"'
        t.value = False
        return t

    def t_MULTI_STRING(self, t):
        r'"(?:[^"]|(?:"")|(?:\\x[0-9a-fA-F]+)|(?:\\.))*"'
        t.lexer.lineno += t.value.count("\n")
        t.value = t.value[1:-1]
        return t

    def t_error(self, t):
        msg = ("Illegal character '{0}' at line {1} col {2}".format(
            t.value[0],
            t.lexer.lineno,
            find_column(t.lexer.lexdata, t.lexpos) + 1))
        if self.error_mode == 'strict':
            exc = SyntaxError(msg)
            exc.lineno = t.lexer.lineno
            raise exc
        t.type = t.value[0]
        t.value = t.value[0]
        t.lexer.skip(1)
        return t

    # yacc Grammar

    def p_xprotocols(self, p):
        """ xprotocols : xprotocols xprotocol
                       | xprotocol
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[2]]

    def p_map_type(self, p):
        """ xprotocol : XPROTOCOL '{' attr_list block_list '}'
            xprotocol : EVAPROTOCOL '{' attr_list block_list '}'
            pipeservice : PIPESERVICE '{' attr_list block_list '}'
            paramfunctor : PARAMFUNCTOR '{' attr_list block_list '}'
            parammap : PARAMMAP '{' attr_list block_list '}'
        """
        p[0] = dict(type=p[1][0],
                    name=p[1][1],
                    attrs=[] if p[3] is None else p[3],
                    value=p[4],
                   )

    def p_str_list_type(self, p):
        """ method : METHOD '{' string_list '}'
            connection : CONNECTION '{' string_list '}'
            event : EVENT '{' string_list '}'
        """
        p[0] = dict(type=p[1][0],
                    name=p[1][1],
                    attrs=[],
                    value=p[3],
                   )

    def p_param_choice(self, p):
        """ paramchoice : PARAMCHOICE '{' attr_list MULTI_STRING '}'
                        | PARAMCHOICE '{' attr_list empty '}'
        """
        p[0] = dict(type=p[1][0],
                    name=p[1][1],
                    attrs=p[3],
                    value=p[4])

    def p_block_list(self, p):
        """ block_list : block_list block
                       | block
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[2]]

    def p_param_array(self, p):
        """ paramarray : PARAMARRAY '{' attr_list curly_lists '}'
                       | PARAMARRAY '{' attr_list '}'
        """
        p[0] = dict(type=p[1][0],
                    name=p[1][1],
                    attrs=p[3],
                    value=p[4] if len(p) == 6 else None)

    def p_curly_lists(self, p):
        """ curly_lists : curly_lists curly_list
                        | curly_list
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[2]]

    def p_curly_lists_empty(self, p):
        """ curly_lists : curly_lists '{' '}'
                        | '{' '}'
        """
        p[0] = [[]] if len(p) == 3 else p[1] + [[]]

    def p_block(self, p):
        """ block : parambool
                  | paramlong
                  | paramdouble
                  | paramstring
                  | paramarray
                  | parammap
                  | paramchoice
                  | paramfunctor
                  | pipeservice
                  | paramcardlayout
                  | evacardlayout
                  | dependency
                  | event
                  | method
                  | connection
        """
        p[0] = p[1]

    def p_basic_type(self, p):
        """ paramstring : PARAMSTRING '{' attr_list empty '}'
                        | PARAMSTRING '{' attr_list MULTI_STRING '}'
            paramdouble : PARAMDOUBLE '{' attr_list empty '}'
                        | PARAMDOUBLE '{' attr_list FLOAT '}'
            paramlong : PARAMLONG '{' attr_list empty '}'
                      | PARAMLONG '{' attr_list INTEGER '}'
            parambool : PARAMBOOL '{' attr_list empty '}'
                      | PARAMBOOL '{' attr_list TRUE '}'
                      | PARAMBOOL '{' attr_list FALSE '}'

        """
        p[0] = dict(type=p[1][0],
                    name=p[1][1],
                    attrs=p[3],
                    value=p[4])

    def p_dependency(self, p):
        """ dependency : DEPENDENCY '{' string_list attr_list '}'
        """
        p[0] = dict(type=p[1][0],
                    name=p[1][1],
                    attrs=[] if p[4] is None else p[4],
                    value=p[3])

    def p_param_card_layout(self, p):
        """ paramcardlayout : PARAMCARDLAYOUT '{' repr controls lines '}'
        """
        p[0] = dict(type=p[1][0],
                    name=p[1][1],
                    attrs=[],
                    value=dict(repr=p[3],
                               controls=p[4],
                               lines=p[5])
                   )

    def p_eva_card_layout(self, p):
        """ evacardlayout : EVACARDLAYOUT '{' MULTI_STRING INTEGER eva_controls lines '}'
                          | EVACARDLAYOUT '{' MULTI_STRING INTEGER eva_controls empty '}'
        """
        # This appears to be the predecessor of ParamCardLayout
        p[0] = dict(type=p[1][0],
                    name=p[1][1],
                    attrs=[],
                    value=dict(repr=p[3],
                               n_controls=p[4],
                               controls=p[5],
                               lines=[] if p[6] is None else p[6])
                   )

    def p_attr_list(self, p):
        """ attr_list : attr_list key_value
                      | key_value
                      |
        """
        if len(p) == 1:  # empty
            p[0] = []
        elif len(p) == 2:  # tagged params or key_value
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_key_value(self, p):
        """key_value : evastringtable
                     | TAG curly_list
                     | TAG scalar
                     | TAG block
        """
        p[0] = p[1] if len(p) == 2 else (p[1][0], p[2])

    def p_scalar(self, p):
        """scalar : FLOAT
                  | INTEGER
                  | FALSE
                  | TRUE
                  | MULTI_STRING
        """
        p[0] = p[1]

    def p_curly_list(self, p):
        """ curly_list : '{' attr_list string_list '}'
                       | '{' attr_list integer_list '}'
                       | '{' attr_list float_list '}'
                       | '{' attr_list bool_list '}'
                       | '{' attr_list curly_lists '}'
        """
        # TODO: Is it okay to just ignore the attr_list here? Seems like the
        #       intention is to allow attribute of the parent array to be
        #       overridden.
        p[0] = p[3]

    def p_scalar_lists(self, p):
        """ string_list : string_list MULTI_STRING
                        | MULTI_STRING
            integer_list : integer_list INTEGER
                         | INTEGER
            float_list : float_list FLOAT
                       | FLOAT
            bool_list : bool_list TRUE
                      | bool_list FALSE
                      | TRUE
                      | FALSE
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[2]]

    def p_controls(self, p):
        """ controls : controls control
                     | control
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[2]]

    def p_eva_controls(self, p):
        """ eva_controls : eva_controls eva_control
                         | eva_control
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[2]]

    def p_lines(self, p):
        """ lines : lines line
                  | line
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[2]]

    def p_control(self, p):
        """ control : CONTROL '{' param pos repr '}'
                    | CONTROL '{' param pos empty '}'
        """
        p[0] = dict(param=p[3],
                    pos=p[4],
                    repr=p[5])

    def p_eva_control(self, p):
        """ eva_control : MULTI_STRING INTEGER INTEGER MULTI_STRING
        """
        p[0] = dict(param=p[1],
                    pos=[p[2], p[3]],
                    repr=p[4])

    def p_eva_string_table(self, p):
        """ evastringtable : EVASTRINGTABLE '{' INTEGER int_strings '}'
        """
        p[0] = (p[1][0], (p[3], p[4]))

    def p_int_strings(self, p):
        """ int_strings : int_strings int_string
                        | int_string
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[2]]

    def p_int_string(self, p):
        """ int_string : INTEGER MULTI_STRING """
        p[0] = (p[1], p[2])

    def p_param(self, p):
        """param : PARAM MULTI_STRING
        """
        p[0] = p[2]

    def p_repr(self, p):
        """repr : REPR MULTI_STRING
        """
        p[0] = p[2]

    def p_pos(self, p):
        """pos : POS INTEGER INTEGER
        """
        p[0] = [p[2], p[3]]

    def p_line(self, p):
        """line : LINE '{' INTEGER INTEGER INTEGER INTEGER '}'
        """
        p[0] = [p[3], p[4], p[5], p[6]]

    def p_empty(self, p):
        'empty :'

    def p_error(self, p):
        if not p:
            msg = "Syntax error at EOF"
        else:
            in_data = p.lexer.lexdata
            msg = ("Syntax error at '{0}', line {1}, col {2}".format(
                p.value, p.lineno, find_column(in_data, p.lexpos) + 1) +
                "\nLine is: '{0}'".format(in_data.splitlines()[p.lineno-1]))
        if self.error_mode == 'strict':
            exc = SyntaxError(msg)
            if not p:
                exc.lineno = -1
            else:
                exc.lineno = p.lineno
            raise exc
        print(msg)

    def reset(self):
        """ Reset lexer ready for new read """
        self.lexer.lineno = 1

    def parse(self, in_str):
        """ Parse `in_str` with XProtocol parser
        """
        self.reset()
        return self.parser.parse(in_str, lexer=self.lexer)


DBL_QUOTE_RE = re.compile(r'(?<!")""(?!")')


def strip_twin_quote(in_str):
    """ Replaces two double quotes together with one double quote

    Does so safely so that triple double quotes not touched.
    """
    return DBL_QUOTE_RE.sub('"', in_str)


ASCCONV_RE = re.compile(
    r'(.*?)### ASCCONV BEGIN ###$\n(.*?)\n^### ASCCONV END ###',
    flags=re.M | re.S)


def split_ascconv(in_str):
    """ Split input string into xprotocol and ASCCONV
    """
    return ASCCONV_RE.match(in_str).groups()


XPROTOCOL_SYMBOLS = XProtocolSymbols()
parse = XPROTOCOL_SYMBOLS.parse
