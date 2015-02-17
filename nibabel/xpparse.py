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
    # Known basic tag identifiers
    basic_tag_ids = {'XProtocol': 'XPROTOCOL',
                     'Class': 'CLASS',
                     'Dll': 'DLL',
                     'Control': 'CONTROL',
                     'Param': 'PARAM',
                     'Pos': 'POS',
                     'Repr': 'REPR',
                     'Line': 'LINE',
                     'Context': 'CONTEXT',
                     'EVAStringTable': 'EVASTRINGTABLE',
                     'Name': 'NAME',
                     'ID': 'ID',
                     'Userversion': 'USERVERSION'}

    # Known tag identifiers with defined types
    typed_tag_ids = {'ParamBool': 'PARAMBOOL',
                     'ParamLong': 'PARAMLONG',
                     'ParamDouble': 'PARAMDOUBLE',
                     'ParamString': 'PARAMSTRING',
                     'ParamArray': 'PARAMARRAY',
                     'ParamMap': 'PARAMMAP',
                     'ParamChoice': 'PARAMCHOICE',
                     'ParamFunctor': 'PARAMFUNCTOR',
                     'ParamCardLayout': 'PARAMCARDLAYOUT',
                     'PipeService': 'PIPESERVICE',
                     'EVACardLayout': 'EVACARDLAYOUT',
                     'Connection': 'CONNECTION',
                     'Dependency': 'DEPENDENCY',
                     'Event': 'EVENT',
                     'Method': 'METHOD'}

    tokens = [
        'TAG',
        'TYPED_TAG',
        'WHITESPACE',
        'INTEGER',
        'FLOAT',
        'MULTI_STRING',
        'TRUE',
        'FALSE',
    ] + list(basic_tag_ids.values()) + list(typed_tag_ids.values())

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

    # Basic tag
    def t_TAG(self, t):
        r'<(?P<tagname>[A-Za-z_][\w_]*)>'
        t.value = t.lexer.lexmatch.group('tagname')
        t.type = self.basic_tag_ids.get(t.value, 'TAG')
        return t

    # Typed tag
    def t_TYPED_TAG(self, t):
        r'<(?P<tagtype>[A-Za-z_][\w_]*)\."(?P<tagname>.*?)">'
        match = t.lexer.lexmatch
        t.value = match.group('tagname')
        t.type = self.typed_tag_ids.get(match.group('tagtype'), 'TYPED_TAG')
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

    def p_xprotocol(self, p):
        """ xprotocol : XPROTOCOL '{' xp_hdr block_list param_cards depends '}'
                      | XPROTOCOL '{' xp_hdr block_list eva_cards depends '}'
                      | XPROTOCOL '{' xp_hdr block_list empty depends '}'
                      | XPROTOCOL '{' xp_hdr block_list param_cards empty '}'
                      | XPROTOCOL '{' xp_hdr block_list eva_cards empty '}'
                      | XPROTOCOL '{' xp_hdr block_list empty empty '}'
        """
        p[0] = dict(type='xprotocol',
                    blocks=p[4],
                    cards=[] if p[5] is None else p[5],
                    depends=[] if p[6] is None else p[6])
        p[0].update(p[3])

    def p_xp_hdr(self, p):
        """ xp_hdr : xp_hdr xp_hdr_key
                   | xp_hdr_key
        """
        if len(p) == 2:
            p[0] = dict([p[1]])
        else:
            p[0] = p[1]
            p[0].update(dict([p[2]]))

    def p_xp_hdr_key(self, p):
        """ xp_hdr_key : name
                       | id
                       | user_version
                       | eva_string_table
        """
        p[0] = p[1]

    def p_name(self, p):
        """ name : NAME MULTI_STRING
        """
        p[0] = ('name', p[2])

    def p_id(self, p):
        """ id : ID INTEGER
        """
        p[0] = ('id', p[2])

    def p_user_version(self, p):
        """ user_version : USERVERSION FLOAT
        """
        p[0] = ('user_version', p[2])

    def p_depends(self, p):
        """ depends : depends dependency
                    | dependency
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_cards(self, p):
        """ param_cards : param_cards param_card_layout
                        | param_card_layout
            eva_cards   : eva_cards eva_card_layout
                        | eva_card_layout
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_pipe_service(self, p):
        """ pipe_service : PIPESERVICE '{' class block_list '}'
        """
        p[0] = {'type': 'pipe_service',
                'name': p[1],
                'class': p[3],
                'value': p[4]}

    def p_param_functor(self, p):
        """ param_functor : PARAMFUNCTOR '{' class block_list emc '}'
        """
        p[0] = {'type': 'param_functor',
                'name': p[1],
                'class': p[3],
                'value': p[4]}
        for param in p[5]:
            key = param['type']
            p[0][key] = param

    def p_param_emc(self, p):
        """ emc : event method connection
                | event connection method
                | method event connection
                | method connection event
                | connection event method
                | connection method event
        """
        p[0] = [p[1], p[2], p[3]]

    def p_method(self, p):
        """ method : METHOD '{' string_list '}'
        """
        p[0] = dict(type='method',
                    name=p[1],
                    args=p[3])

    def p_connection(self, p):
        """ connection : CONNECTION '{' string_list '}'
        """
        p[0] = dict(type='connection',
                    name=p[1],
                    args=p[3])

    def p_event(self, p):
        """ event : EVENT '{' string_list '}'
        """
        p[0] = dict(type='event',
                    name=p[1],
                    args=p[3])

    def p_param_choice(self, p):
        """ param_choice : PARAMCHOICE '{' attr_list MULTI_STRING '}'
                         | PARAMCHOICE '{' attr_list empty '}'
        """
        p[0] = dict(type='param_choice',
                    name=p[1],
                    attrs=p[3],
                    value=p[4])

    def p_param_map(self, p):
        """ param_map : PARAMMAP '{' block_list '}'
        """
        p[0] = dict(type='param_map',
                    name=p[1],
                    value=p[3])

    def p_block_list(self, p):
        """ block_list : block_list block
                       | block
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[2]]

    def p_param_array(self, p):
        """ param_array : PARAMARRAY '{' attr_list curly_lists '}'
        """
        p[0] = dict(type='param_array',
                    name=p[1],
                    attrs=p[3],
                    value=p[4])

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
        """ block : param_bool
                  | param_long
                  | param_double
                  | param_string
                  | param_array
                  | param_map
                  | param_choice
                  | param_functor
                  | pipe_service
        """
        p[0] = p[1]

    def p_param_string(self, p):
        """ param_string : PARAMSTRING '{' attr_list empty '}'
                         | PARAMSTRING '{' attr_list MULTI_STRING '}'
        """
        p[0] = dict(type='param_string',
                    name=p[1],
                    attrs=p[3],
                    value=p[4])

    def p_param_double(self, p):
        """ param_double : PARAMDOUBLE '{' attr_list empty '}'
                         | PARAMDOUBLE '{' attr_list FLOAT '}'
        """
        p[0] = dict(type='param_double',
                    name=p[1],
                    attrs=p[3],
                    value=p[4])

    def p_param_long(self, p):
        """ param_long : PARAMLONG '{' attr_list empty '}'
                       | PARAMLONG '{' attr_list INTEGER '}'
        """
        p[0] = dict(type='param_long',
                    name=p[1],
                    attrs=p[3],
                    value=p[4])

    def p_param_bool(self, p):
        """ param_bool : PARAMBOOL '{' attr_list empty '}'
                       | PARAMBOOL '{' attr_list TRUE '}'
                       | PARAMBOOL '{' attr_list FALSE '}'
        """
        p[0] = dict(type='param_bool',
                    name=p[1],
                    attrs=p[3],
                    value=p[4])

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
        """key_value : TAG curly_list
                     | TAG scalar
                     | TAG block
        """
        p[0] = (p[1], p[2])

    def p_scalar(self, p):
        """scalar : FLOAT
                  | INTEGER
                  | FALSE
                  | TRUE
                  | MULTI_STRING
        """
        p[0] = p[1]

    def p_dependency(self, p):
        """ dependency : DEPENDENCY '{' string_list empty empty '}'
                       | DEPENDENCY '{' string_list dll empty '}'
                       | DEPENDENCY '{' string_list dll context '}'
                       | DEPENDENCY '{' string_list empty context '}'
        """
        p[0] = dict(type='dependency',
                    name=p[1],
                    values=p[3],
                    dll=p[4],
                    context=p[5])

    def p_curly_list(self, p):
        """ curly_list : '{' string_list '}'
                       | '{' integer_list '}'
                       | '{' float_list '}'
                       | '{' bool_list '}'
        """
        p[0] = p[2]

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

    def p_param_card_layout(self, p):
        """ param_card_layout : PARAMCARDLAYOUT '{' repr controls lines '}'
        """
        p[0] = dict(type='param_card_layout',
                    name=p[1],
                    repr=p[3],
                    controls=p[4],
                    lines=p[5])

    def p_eva_card_layout(self, p):
        """ eva_card_layout : EVACARDLAYOUT '{' MULTI_STRING INTEGER eva_controls lines '}'
        """
        # This appears to be the predecessor of ParamCardLayout
        p[0] = dict(type='eva_card_layout',
                    name=p[1],
                    repr=p[3],
                    n_controls=p[4],
                    controls=p[5],
                    lines=p[6])

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
        """ eva_string_table : EVASTRINGTABLE '{' INTEGER int_strings '}'
        """
        p[0] = (p[1], (p[3], p[4]))

    def p_int_strings(self, p):
        """ int_strings : int_strings int_string
                        | int_string
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[2]]

    def p_int_string(self, p):
        """ int_string : INTEGER MULTI_STRING """
        p[0] = (p[1], p[2])

    def p_class(self, p):
        """class : CLASS MULTI_STRING
        """
        p[0] = p[2]

    def p_context(self, p):
        """context : CONTEXT MULTI_STRING
        """
        p[0] = p[2]

    def p_dll(self, p):
        """dll : DLL MULTI_STRING
        """
        p[0] = p[2]

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
