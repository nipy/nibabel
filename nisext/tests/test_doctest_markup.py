""" Testing doctest markup tests
"""

import sys
from ..py3builder import doctest_markup, byter

from numpy.testing import (assert_array_almost_equal, assert_array_equal, dec)

from nose.tools import assert_true, assert_equal, assert_raises

is_2 = sys.version_info[0] < 3
skip42 = dec.skipif(is_2)

# Tell 23dt processing to pass the rest of this file unchanged.  We don't want
# the processor to mess up the example string
#23dt skip rest

IN_TXT = """

Anonymous lines, also blanks

As all that is empty, use entropy, and endure

# Comment, unchanged

#23dt comment not processed without doctest marker
>>> #23dthere: no whitespace; comment not recognized even as error
>>>#23dt nor without preceding whitespace
>>> #23dt not correct syntax creates error
>>> #23dt novar: 'undefined variable creates error'
>>> #23dt here: 'OK'
>>> #23dt here    :    'tolerates whitespace'
>>> #23dt here + 0 : 'OK'
>>> #23dt here -0   : 'OK'
>>> #23dt here - here + here + 0: 'OK'
>>> #23dt here *0   : 'only allowed plus or minus'
>>> #23dt : 'empty means here'
>>> #23dt   : 'regardless of whitespace'
>>> #23dt 'need colon'
>>> #23dt here : 3bad syntax
>>> #23dt here : 1/0
>>> #23dt next : line.replace('some','')
something
>>> #23dt next : replace('some','')
something
>>> #23dt next : lines[next].replace('some','')
something
>>> #23dt next + 1: line.replace('some','')
something
something
>>> #23dt next : lines[next+1].replace('some','')
this is the line where replacement happens
something
  >>>  whitespace     #23dt : 'OK'
>>> from io import StringIO as BytesIO #23dt : replace('StringIO as ', '')
>>> from io import StringIO #23dt : BytesIO
>>> from io import StringIO #23dt :  BytesIO
"""

OUT_TXT = """

Anonymous lines, also blanks

As all that is empty, use entropy, and endure

# Comment, unchanged

#23dt comment not processed without doctest marker
>>> #23dthere: no whitespace; comment not recognized even as error
>>>#23dt nor without preceding whitespace
>>> #23dt not correct syntax creates error
>>> #23dt novar: 'undefined variable creates error'
>>> OK#23dt here: 'OK'
>>> tolerates whitespace#23dt here    :    'tolerates whitespace'
>>> OK#23dt here + 0 : 'OK'
>>> OK#23dt here -0   : 'OK'
>>> OK#23dt here - here + here + 0: 'OK'
>>> #23dt here *0   : 'only allowed plus or minus'
>>> empty means here#23dt : 'empty means here'
>>> regardless of whitespace#23dt   : 'regardless of whitespace'
>>> #23dt 'need colon'
>>> #23dt here : 3bad syntax
>>> #23dt here : 1/0
>>> #23dt next : line.replace('some','')
thing
>>> #23dt next : replace('some','')
thing
>>> #23dt next : lines[next].replace('some','')
thing
>>> #23dt next + 1: line.replace('some','')
something
thing
>>> #23dt next : lines[next+1].replace('some','')
thing
something
  >>>  OK     #23dt : 'OK'
>>> from io import BytesIO #23dt : replace('StringIO as ', '')
>>> from io import BytesIO #23dt : BytesIO
>>> from io import BytesIO #23dt :  BytesIO
"""

ERR_TXT = \
""">>> #23dt not correct syntax creates error
>>> #23dt novar: 'undefined variable creates error'
>>> #23dt here *0   : 'only allowed plus or minus'
>>> #23dt 'need colon'
>>> #23dt here : 3bad syntax
>>> #23dt here : 1/0
"""

def test_some_text():
    out_lines, err_tuples = doctest_markup(IN_TXT.splitlines(True))
    assert_equal(out_lines, OUT_TXT.splitlines(True))
    err_lines, err_msgs = zip(*err_tuples)
    assert_equal(list(err_lines), ERR_TXT.splitlines(True))


IN_BYTES_TXT = """\

Phatos lives

>>> 'hello' #23dt : bytes
>>> (1, 'hello') #23dt : bytes
>>> 'hello' #23dt next : bytes
'TRACK'
>>> ('hello', 1, 'world') #23dt : bytes
>>> 3bad_syntax #23dt : bytes
"""

OUT_BYTES_TXT = """\

Phatos lives

>>> b'hello' #23dt : bytes
>>> (1, b'hello') #23dt : bytes
>>> 'hello' #23dt next : bytes
b'TRACK'
>>> (b'hello', 1, b'world') #23dt : bytes
>>> 3bad_syntax #23dt : bytes
"""

ERR_BYTES_TXT = \
""">>> 3bad_syntax #23dt : bytes
"""

@skip42
def test_bytes_text():
    out_lines, err_tuples = doctest_markup(IN_BYTES_TXT.splitlines(True))
    assert_equal(out_lines, OUT_BYTES_TXT.splitlines(True))
    err_lines, err_msgs = zip(*err_tuples)
    assert_equal(list(err_lines), ERR_BYTES_TXT.splitlines(True))


@skip42
def test_byter():
    # Test bytes formatter
    assert_equal('(b"hello \' world", b\'again\')',
                 byter('("hello \' world", "again")'))
    line = "_ = bio.write(' ' * 10)"
    assert_equal(
        byter(line),
        "_ = bio.write(b' ' * 10)")

