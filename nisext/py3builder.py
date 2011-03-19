""" distutils utilities for porting to python 3 within 2-compatible tree """

from __future__ import with_statement
import re
import ast
from . import codegen

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
        out = doctest_markup(res)
        with open(fname, 'wt') as fobj:
            fobj.write(''.join(out))


REGGIES = (
     (re.compile('from\s+io\s+import\s+StringIO\s+as\s+BytesIO\s*?(?=$)'),
     'from io import BytesIO'),
)

MARK_COMMENT = re.compile('(\s*>>>\s+)(.*?)(\s*#2to3:\s*)(.*?\s*)$', re.DOTALL)
PLACE_EXPR = re.compile('\s*([\w+\- ]+);\s*(.*)$')
INDENT_SPLITTER = re.compile('(\s*)(.*?)(\s*)$', re.DOTALL)

def doctest_markup(in_lines):
    """ Process doctest comment markup on sequence of strings

    The markup is very crude.  The algorithm looks for lines starting with
    ``>>>``.  All other lines are passed through unchanged.

    Next it looks for known simple search replaces and does those.

    * ``from StringIO import StringIO as BytesIO`` replaced by ``from io import
      BytesIO``.

    Next it looks to see if the line ends with a comment starting with ``#2to3:``.

    The stuff after the ``#2to3:`` marker is, of course, a little language, of
    form <place>; <expr>

    * <place> is an expression giving a line number.  In this expression, ``here`` is
      a variable referring to the current line number, and ``next`` is just
      ``here+1``.
    * <expr> is a python3 expression returning a processed value, where
      ``line`` contains the line referred to by line number ``here``, and
      ``lines`` is a list of all lines. ``lines[here]`` gives the value of
      ``line``.

    An <expr> beginning with "replace(" we take to be short for "line.replace(".

    If <expr> is just 'bytes', then make all the strings in the selected line be
    byte strings.

    Parameters
    ----------
    in_lines : sequence of str

    Returns
    -------
    out_lines : sequence of str

    Examples
    --------
    The next three lines all do the same thing:

    >> a = '1234567890' #2to3: here; line.replace("'12", "b'12")
    >> a = '1234567890' #2to3: here; replace("'12", "b'12")
    >> a = '1234567890' #2to3: here; bytes

    and that is to give the output (e.g):

    >> a = b'1234567890' #2to3: here; line.replace("'12", "b'12")

    in the processed doctest.

    You might want to process the line after the comment - such as test output.
    The next test replaces "'a string'" with "b'a string'"

    >> 'a string'.encode('ascii') #2to3: next; bytes
    'a string'

    This might work too, to do the same thing:

    >> 'a string'.encode('ascii') #2to3: here+1; bytes
    'a string'
    """
    pos = 0
    lines = list(in_lines)
    while pos < len(lines):
        this = lines[pos]
        here = pos
        pos += 1
        if not this.lstrip().startswith('>>>'):
            continue
        # Check simple regexps (no markup)
        for reg, substr in REGGIES:
            if reg.search(this):
                lines[here] = reg.sub(substr, this)
                break
        # Check for specific markup
        mark_match = MARK_COMMENT.search(this)
        if mark_match is None:
            continue
        docbits, start, marker, markup = mark_match.groups()
        pe_match = PLACE_EXPR.match(markup)
        if pe_match is None:
            continue
        # Get place, expr expressions
        place, expr = pe_match.groups()
        try:
            place = eval(place, {'here': here, 'next': here+1})
        except:
            print('Error finding place with "%s", line "%s"; skipping' %
                    (place, this))
            continue
        # Prevent processing operating on 2to3 comment part of line
        if place == here:
            line = start
        else:
            line = lines[place]
        # Process expr
        expr = expr.strip()
        # Shorthand
        if expr == 'bytes':
            # Any strings on the given line are byte strings
            pre, mid, post = INDENT_SPLITTER.match(line).groups()
            res = byter(mid)
            res = pre + res + post
        else:
            # If expr starts with 'replace', implies "line.replace"
            if expr.startswith('replace('):
                expr = 'line.' + expr
            try:
                res = eval(expr, dict(line=line,
                                    lines=lines))
            except:
                print('Error working on "%s" at line %d with "%s"; skipping' %
                    (line, place, expr))
                continue
        # Put back comment if removed
        if place == here:
            res = docbits + res + marker + markup
        if res != line:
            lines[place] = res
    return lines


class RewriteStr(ast.NodeTransformer):
    def visit_Str(self, node):
        return ast.Bytes(node.s.encode('ascii'))


def byter(src):
    """ Convert strings in `src` to byte string literals
    """
    tree = ast.parse(src)
    tree = RewriteStr().visit(tree)
    return codegen.to_source(tree)

