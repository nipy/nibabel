""" Test printable table
"""
from __future__ import division, print_function

import sys
import numpy as np

from ..rstutils import rst_table

from nose import SkipTest
from nose.tools import assert_equal, assert_raises


def test_rst_table():
    # Tests for printable table function
    R, C = 3, 4
    cell_values = np.arange(R * C).reshape((R, C))
    if (sys.version_info[:3] == (3, 2, 3) and np.__version__ == '1.6.1'):
        raise SkipTest("Known (later fixed) bug in python3.2/numpy "
                       "treating np.int64 as str")
    assert_equal(rst_table(cell_values),
                 """+--------+--------+--------+--------+--------+
|        | col[0] | col[1] | col[2] | col[3] |
+========+========+========+========+========+
| row[0] |  0.00  |  1.00  |  2.00  |  3.00  |
| row[1] |  4.00  |  5.00  |  6.00  |  7.00  |
| row[2] |  8.00  |  9.00  | 10.00  | 11.00  |
+--------+--------+--------+--------+--------+"""
                 )
    assert_equal(rst_table(cell_values, ['a', 'b', 'c']),
                 """+---+--------+--------+--------+--------+
|   | col[0] | col[1] | col[2] | col[3] |
+===+========+========+========+========+
| a |  0.00  |  1.00  |  2.00  |  3.00  |
| b |  4.00  |  5.00  |  6.00  |  7.00  |
| c |  8.00  |  9.00  | 10.00  | 11.00  |
+---+--------+--------+--------+--------+"""
                 )
    assert_raises(ValueError,
                  rst_table, cell_values, ['a', 'b'])
    assert_raises(ValueError,
                  rst_table, cell_values, ['a', 'b', 'c', 'd'])
    assert_equal(rst_table(cell_values, None, ['1', '2', '3', '4']),
                 """+--------+-------+-------+-------+-------+
|        |   1   |   2   |   3   |   4   |
+========+=======+=======+=======+=======+
| row[0] |  0.00 |  1.00 |  2.00 |  3.00 |
| row[1] |  4.00 |  5.00 |  6.00 |  7.00 |
| row[2] |  8.00 |  9.00 | 10.00 | 11.00 |
+--------+-------+-------+-------+-------+"""
                 )
    assert_raises(ValueError,
                  rst_table, cell_values, None, ['1', '2', '3'])
    assert_raises(ValueError,
                  rst_table, cell_values, None, list('12345'))
    assert_equal(rst_table(cell_values, title='A title'),
                 """*******
A title
*******

+--------+--------+--------+--------+--------+
|        | col[0] | col[1] | col[2] | col[3] |
+========+========+========+========+========+
| row[0] |  0.00  |  1.00  |  2.00  |  3.00  |
| row[1] |  4.00  |  5.00  |  6.00  |  7.00  |
| row[2] |  8.00  |  9.00  | 10.00  | 11.00  |
+--------+--------+--------+--------+--------+"""
                 )
    assert_equal(rst_table(cell_values, val_fmt='{0}'),
                 """+--------+--------+--------+--------+--------+
|        | col[0] | col[1] | col[2] | col[3] |
+========+========+========+========+========+
| row[0] | 0      | 1      | 2      | 3      |
| row[1] | 4      | 5      | 6      | 7      |
| row[2] | 8      | 9      | 10     | 11     |
+--------+--------+--------+--------+--------+"""
                 )
    # Doing a fancy cell format
    cell_values_back = np.arange(R * C)[::-1].reshape((R, C))
    cell_3d = np.dstack((cell_values, cell_values_back))
    assert_equal(rst_table(cell_3d, val_fmt='{0[0]}-{0[1]}'),
                 """+--------+--------+--------+--------+--------+
|        | col[0] | col[1] | col[2] | col[3] |
+========+========+========+========+========+
| row[0] | 0-11   | 1-10   | 2-9    | 3-8    |
| row[1] | 4-7    | 5-6    | 6-5    | 7-4    |
| row[2] | 8-3    | 9-2    | 10-1   | 11-0   |
+--------+--------+--------+--------+--------+"""
                 )
    # Test formatting characters
    formats = dict(
        down='!',
        along='_',
        thick_long='~',
        cross='%',
        title_heading='#')
    assert_equal(rst_table(cell_values, title='A title', format_chars=formats),
                 """#######
A title
#######

%________%________%________%________%________%
!        ! col[0] ! col[1] ! col[2] ! col[3] !
%~~~~~~~~%~~~~~~~~%~~~~~~~~%~~~~~~~~%~~~~~~~~%
! row[0] !  0.00  !  1.00  !  2.00  !  3.00  !
! row[1] !  4.00  !  5.00  !  6.00  !  7.00  !
! row[2] !  8.00  !  9.00  ! 10.00  ! 11.00  !
%________%________%________%________%________%"""
                 )
    formats['funny_value'] = '!'
    assert_raises(ValueError,
                  rst_table,
                  cell_values, title='A title', format_chars=formats)
    return
