""" distutils utilities for porting to python 3 within 2-compatible tree """

from __future__ import division, print_function, absolute_import

import sys
import re

try:
    from distutils.command.build_py import build_py_2to3
except ImportError:
    # 2.x - no parsing of code
    from distutils.command.build_py import build_py
else: # Python 3
    # Command to also apply 2to3 to doctests
    from distutils import log
    class build_py(build_py_2to3):
        def run_2to3(self, files):
            # Add doctest parsing; this stuff copied from distutils.utils in
            # python 3.2 source
            if not files:
                return
            fixer_names, options, explicit = (self.fixer_names,
                                              self.options,
                                              self.explicit)
            # Make this class local, to delay import of 2to3
            from lib2to3.refactor import RefactoringTool, get_fixers_from_package
            class DistutilsRefactoringTool(RefactoringTool):
                def log_error(self, msg, *args, **kw):
                    log.error(msg, *args)

                def log_message(self, msg, *args):
                    log.info(msg, *args)

                def log_debug(self, msg, *args):
                    log.debug(msg, *args)

            if fixer_names is None:
                fixer_names = get_fixers_from_package('lib2to3.fixes')
            r = DistutilsRefactoringTool(fixer_names, options=options)
            r.refactor(files, write=True)
            # Then doctests
            r.refactor(files, write=True, doctests_only=True)
            # Then custom doctests markup
            doctest_markup_files(files)


def doctest_markup_files(fnames):
    """ Process simple doctest comment markup on sequence of filenames

    Parameters
    ----------
    fnames : seq
        sequence of filenames

    Returns
    -------
    None
    """
    for fname in fnames:
        with open(fname, 'rt') as fobj:
            res = list(fobj)
        out, errs = doctest_markup(res)
        for err_tuple in errs:
            print('Marked line %s unchanged because "%s"' % err_tuple)
        with open(fname, 'wt') as fobj:
            fobj.write(''.join(out))


MARK_COMMENT = re.compile('(\s*>>>\s+)(.*?)(\s*#23dt\s+)(.*?\s*)$', re.DOTALL)
PLACE_LINE_EXPRS = re.compile('\s*([\w+\- ]*):\s*(.*)$')
INDENT_SPLITTER = re.compile('(\s*)(.*?)(\s*)$', re.DOTALL)

def doctest_markup(in_lines):
    """ Process doctest comment markup on sequence of strings

    The algorithm looks for lines that start with optional whitespace followed
    by ``>>>`` and ending with a comment starting with ``#23dt``.  The stuff
    after the ``#23dt`` marker is the *markup* and gives instructions for
    modifying the corresponding line or some other line.

    The *markup* is of form <place-expr> : <line-expr>.  Let's say the output
    lines are in a variable ``out_lines``.

    * <place-expr> is an expression giving a line number.  In this expression,
      the two variables defined are ``here`` (giving the current line number),
      and ``next == here+1``.  Let's call the result of <place-expr> ``place``.
      If <place-expr> is empty (only whitespace before the colon) then ``place
      == here``. The result of <line-expr> will replace ``lines[place]``.
    * <line-expr> is a special value (see below) or a python3 expression
      returning a processed value, where ``line`` contains the line referred to
      by line number ``place``, and ``lines`` is a list of all lines.  If
      ``place != here``, then ``line == lines[place]``.  If ``place == here``
      then ``line`` will be the source line, minus the comment and markup.

    A <line-expr> beginning with "replace(" we take to be short for
    "line.replace(".

    Special values; if <line-expr> ==:

    * 'bytes': make all the strings in the selected line be byte strings. This
      algormithm uses the ``ast`` module, so the text in which it works must be
      valid python 3 syntax.
    * 'BytesIO': shorthand for ``replace('StringIO', 'BytesIO')``

    There is also a special non-doctest comment markup - '#23dt skip rest'.  If
    we find that comment (with whitespace before or after) as a line in the
    file, we just pass the rest of the file unchanged.  This is a hack to stop
    23dt processing its own tests.

    Parameters
    ----------
    in_lines : sequence of str

    Returns
    -------
    out_lines : sequence of str
        lines with processing applied
    error_tuples : sequence of (str, str)
        sequence of 2 element tuples, where the first entry in the tuple is one
        line that generated an error during processing, and the second is the
        explanatory message for the error.  These lines remain unchanged in
        `out_lines`.

    Examples
    --------
    The next three lines all do the same thing:

    >> a = '1234567890' #23dt here: line.replace("'12", "b'12")
    >> a = '1234567890' #23dt here: replace("'12", "b'12")
    >> a = '1234567890' #23dt here: bytes

    and that is to result in the part before the comment changing to:

    >> a = b'1234567890'

    The part after the comment (including markup) stays the same.

    You might want to process the line after the comment - such as test output.
    The next test replaces "'a string'" with "b'a string'"

    >> 'a string'.encode('ascii') #23dt next: bytes
    'a string'

    This might work too, to do the same thing:

    >> 'a string'.encode('ascii') #23dt here+1: bytes
    'a string'
    """
    out_lines = list(in_lines)[:]
    err_tuples = []
    for pos, this in enumerate(out_lines):
        # Check for 'leave the rest' markup
        if this.strip() == '#23dt skip rest':
            break
        # Check for docest line with markup
        mark_match = MARK_COMMENT.search(this)
        if mark_match is None:
            continue
        docbits, marked_line, marker, markup = mark_match.groups()
        place_line_match = PLACE_LINE_EXPRS.match(markup)
        if place_line_match is None:
            msg = ('Found markup "%s" in line "%s" but wrong syntax' %
                   (markup, this))
            err_tuples.append((this, msg))
            continue
        place_expr, line_expr = place_line_match.groups()
        exec_globals = {'here': pos, 'next': pos+1}
        if place_expr.strip() == '':
            place = pos
        else:
            try:
                place = eval(place_expr, exec_globals)
            except:
                msg = ('Error finding place with "%s" in line "%s"' %
                    (place_expr, this))
                err_tuples.append((this, msg))
                continue
        # Prevent processing operating on 23dt comment part of line
        if place == pos:
            line = marked_line
        else:
            line = out_lines[place]
        # Shorthand
        if line_expr == 'bytes':
            # Any strings on the given line are byte strings
            pre, mid, post = INDENT_SPLITTER.match(line).groups()
            try:
                res = byter(mid)
            except:
                err = sys.exc_info()[1]
                msg = ('Error "%s" parsing "%s"' % (err, err))
                err_tuples.append((this, msg))
                continue
            res = pre + res + post
        else:
            exec_globals.update({'line': line, 'lines': out_lines})
            # If line_expr starts with 'replace', implies "line.replace"
            if line_expr.startswith('replace('):
                line_expr = 'line.' + line_expr
            elif line_expr == 'BytesIO':
                line_expr = "line.replace('StringIO', 'BytesIO')"
            try:
                res = eval(line_expr, exec_globals)
            except:
                err = sys.exc_info()[1]
                msg = ('Error "%s" working on "%s" at line %d with "%s"' %
                       (err, line, place, line_expr))
                err_tuples.append((this, msg))
                continue
        # Put back comment if removed
        if place == pos:
            res = docbits + res + marker + markup
        if res != line:
            out_lines[place] = res
    return out_lines, err_tuples


def byter(src):
    """ Convert strings in `src` to byte string literals

    Parameters
    ----------
    src : str
        source string.  Must be valid python 3 source

    Returns
    -------
    p_src : str
        string with ``str`` literals replace by ``byte`` literals
    """
    import ast
    from . import codegen
    class RewriteStr(ast.NodeTransformer):
        def visit_Str(self, node):
            return ast.Bytes(node.s.encode('ascii'))
    tree = ast.parse(src)
    tree = RewriteStr().visit(tree)
    return codegen.to_source(tree)

