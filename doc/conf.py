# -*- coding: utf-8 -*-
#
# PyNIfTI documentation build configuration file, created by
# sphinx-quickstart on Sun May  4 09:06:06 2008.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# The contents of this file are pickled, so don't put values in the namespace
# that aren't pickleable (module imports are okay, they're removed automatically).
#
# All configuration values have a default value; values that are commented out
# serve to show the default value.

import sys, os, re
import numpy as N
# also import pynifti itself to get the version string
import nibabel

try:
    import matplotlib
    matplotlib.use('svg')
except:
    pass

##################################################
# Config settings are at the bottom of the file! #
##################################################

# If your extensions are in another directory, add it here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#sys.path.append(os.path.abspath('some/directory'))


def extractItemListBlock(blocktypes, lines):
    """Extract a number of lines belonging to an indented block.

    The block is defined by a minimum indentation level, in turn
    defined by a line starting with any string given by the 'blocktypes'
    sequence.

    It returns the lines matching the block and the start and endline index
    wrt the original line sequence.

    WARNING: It may explode if there is more than one block with the same
    identifier.
    """
    param = None
    in_block = False
    indent = None
    start_line = None
    end_line = None

    for i, line in enumerate(lines):
        # ignore empty lines
        if line.isspace() or not len(line.strip()):
            continue

        # strip leading whitespace
        sline = line.lstrip()

        # look for block start
        if N.any([sline.startswith(bt) for bt in blocktypes]):
            in_block = True
            indent = len(line) - len(sline)
            start_line = i
            continue

        # check if end is reached
        if in_block and len(line) - len(sline) <= indent:
            end_line = i
            return param, start_line, end_line

        # store param block line
        if in_block:
            if not param:
                param = []
            param.append(line)

    # when nothing follows param block
    if start_line:
        end_line = len(lines) - 1

    return param, start_line, end_line


def smoothName(s):
    """Handle all kinds of voodoo cases, that might disturb RsT
    """
    s = s.strip()
    s = re.sub('\*', '\*', s)
    return s


def segmentItemList(lines, name):
    """Parse the lines of a block into segment items of the format
    used in PyMVPA::

      name[: type]
        (multiline) description

    """
    # assumes no empty lines left!
    items = []
    last_item = None

    # determine indentation level
    indent = len(lines[0]) - len(lines[0].lstrip())

    for line in lines:
        # if top level indent, we have a parameter def
        if indent == len(line) - len(line.lstrip()):
            # rescue previous one
            if last_item is not None:
                items.append(last_item)
                last_item = None

            last_item = {'name': None, 'type': None, 'descr': []}
            # try splitting param def
            l = line.split(':')
            if len(l) >= 2:
                last_item['name'] = smoothName(l[0])
                last_item['type'] = u':'.join(l[1:]).strip()
            elif len(l) == 1:
                last_item['name'] = smoothName(line)
            else:
                print line
                raise RuntimeError, \
                      'Should not have happend, inspect %s' % name
        else:
            # it must belong to last_item and be its description
            if last_item is None:
                print line
                raise ValueError, \
                      'Parameter description, without parameter in %s' % name
            last_item['descr'].append(line.strip())

    if last_item is not None:
        items.append(last_item)

    return items


def reformatParameterBlock(lines, name):
    """Format a proper parameters block from the lines of a docstring
    version of this block.
    """
    params = segmentItemList(lines, name)

    out = []
    # collection is done, now pretty print
    for p in params:
        out.append(':param ' + p['name'] + ': ')
        if len(p['descr']):
            # append first description line to previous one
            out[-1] += p['descr'][0]
            for l in p['descr'][1:]:
                out.append('  ' + l)
        if p['type']:
            out.append(':type ' + p['name'] + ': ' + p['type'])

    # safety line
    out.append(u'')
    return out


def reformatReturnsBlock(lines, name):
    """Format a proper returns block from the lines of a docstring
    version of this block.
    """
    ret = segmentItemList(lines, name)

    if not len(ret) == 1:
        raise ValueError, \
              '%s docstring specifies more than one return value' % name

    ret  = ret[0]
    out = []
    out.append(':rtype: ' + ret['name'])
    if len(ret['descr']):
        out.append(':returns:')
        for l in ret['descr']:
            out.append('  ' + l)

    # safety line
    out.append(u'')
    return out


def reformatExampleBlock(lines, name):
    """Turn an example block into a verbatim text.
    """
    out = [u'::', u'']
    out += lines
    # safety line
    out.append(u'')
    return out


# demo function to access docstrings for processing
def dumpit(app, what, name, obj, options, lines):
    """ For each docstring this function is called with the following set of
    arguments:

    app
      the Sphinx application object
    what
      the type of the object which the docstring belongs to (one of "module",
      "class", "exception", "function", "method", "attribute")
    name
      the fully qualified name of the object
    obj
      the object itself
    options
      the options given to the directive: an object with attributes
      inherited_members, undoc_members, show_inheritance and noindex that are
      true if the flag option of same name was given to the auto directive
    lines
      the lines of the docstring (as a list)
    """
    param, pstart, pend = extractItemListBlock([':Parameters:',
                                                ':Parameter:'], lines)
    if param:
        # make it beautiful
        param = reformatParameterBlock(param, name)

        # replace old block with new one
        lines[pstart:pend] = param

    returns, rstart, rend = extractItemListBlock([':Returns:'], lines)
    if returns:
        returns = reformatReturnsBlock(returns, name)
        lines[rstart:rend] = returns

    examples, exstart, exend = extractItemListBlock([':Examples:',
                                                     ':Example:'], lines)
    if examples:
        print 'WARNING: Example in %s should become a proper snippet' % name
        examples = reformatExampleBlock(examples, name)
        lines[exstart:exend] = examples

    # kill things that sphinx does not know
    ls, lstart, lend = extractItemListBlock(['.. packagetree::'], lines)
    if ls:
        del(lines[lstart:lend])

    # add empty line at begining of class docs to separate base class list from
    # class docs (should actually be done by sphinx IMHO)
    if what == 'class':
        lines.insert(0, u'')


# make this file a sphinx extension itself, to be able to do docstring
# post-processing
def setup(app):
    app.connect('autodoc-process-docstring', dumpit)







# General configuration
# ---------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.txt'

# The master toctree document.
master_doc = 'contents'

# General substitutions.
project = 'PyNIfTI'
copyright = '2006-2009, Michael Hanke'

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
# The short X.Y version.
version = nibabel.__version__
# The full version, including alpha/beta/rc tags.
release = nibabel.__version__

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
unused_docs = []

# what to put into API doc (just class doc, just init, or both
autoclass_content = 'both'

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# Options for HTML output
# -----------------------

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
html_style = 'pynifti.css'

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = 'Home'

# The name of an image file (within the static path) to place at the top of
# the sidebar.
#html_logo = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
html_sidebars = {'index': 'indexsidebar.html'}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {'index': 'index.html'}

# If false, no module index is generated.
html_use_modindex = False

# If true, the reST sources are included in the HTML build as _sources/<name>.
#html_copy_source = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.
#html_use_opensearch = False

# Output file base name for HTML help builder.
htmlhelp_basename = 'PyNIfTIdoc'


# Options for LaTeX output
# ------------------------

# The paper size ('letter' or 'a4').
latex_paper_size = 'a4'

# The font size ('10pt', '11pt' or '12pt').
latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class [howto/manual]).
latex_documents = [
  ('manual', 'PyNIfTI-Manual.tex', 'PyNIfTI Manual',
   'Michael~Hanke', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = os.path.join('_static', 'logo.pdf')

# Additional stuff for the LaTeX preamble.
latex_preamble = """
\usepackage{enumitem}
\setdescription{style=nextline,font=\\normalfont}
"""

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
latex_use_modindex = True
