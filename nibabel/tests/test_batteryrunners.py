''' Tests for BatterRunner and Report objects

These classes / objects are for generic checking / fixing batteries

The class will run a series of checks, optionally running fixes for
problems found in checks.  

To run checks only, and return problem report objects::

   >>> btrun = BatteryRunner()
   >>> report_seq =  btrun.check_only(obj, checks)

To run checks and fixes, returning fixed object, problem report object, with possible fix messages::

   >>> fixed_obj, report_seq = btrun.check_fix(obj, checks)

Reports are iterable things, where the elements in the iterations are
``Problems``, with attributes ``obj``, ``error``, ``problem_level``,
``problem_msg``, and possibly empty ``fix_msg``.  The ``problem_level`` is an
integer, giving the level of remaining problem, from 0 (no problem) to
50 (very bad problem).  The ``error`` can be one of ``None`` if no error
to suggest, or an Exception class that the user might consider raising
for this sitation.  The ``problem_msg`` and ``fix_msg`` are human
readable strings that should explain what happened.

The checks are a sequence of callables, looking like this::

   >>> report = chk(obj, fix=False)

or::

   >>> report_seq = chk(obj, fix=True)

For example, for the Analyze header, we need to check the datatype::

    def chk_datatype(hdr, fix=True):
        ret = Report(hdr, HeaderDataError)
        code = int(hdr['datatype'])
        try:
            dtype = AnalyzeHeader._data_type_codes.dtype[code]
        except KeyError:
            ret.problem_level = 40
            ret.problem_msg = 'data code not recognized'
        else:
            if dtype.type is np.void:
                ret.problem_level = 40
                ret.problem_msg = 'data code not supported'
        if fix:
            ret.fix_problem_msg = 'not attempting fix'
        return ret

    # or the bitpix

    def chk_bitpix(hdr, fix=True):
        ret = Report(hdr, HeaderDataError)
        code = int(hdr['datatype'])
        try:
            dt = AnalyzeHeader._data_type_codes.dtype[code]
        except KeyError:
            ret.problem_level = 10
            ret.problem_msg = 'no valid datatype to fix bitpix'
        bitpix = dt.itemsize * 8
        ret = Report(hdr)
        if bitpix == hdr['bitpix']:
            return ret
        ret.problem_msg = 'bitpix does not match datatype')
        if not fix:
            ret.problem_level = 10
            return ret
        hdr['bitpix'] = bitpix # inplace modification
        ret.problem_level = 0
        ret.fix_msg = 'setting bitpix to match datatype'
        return ret

    # or the pixdims

    def chk_pixdims(hdr, fix=True):
        ret = Report(hdr, HeaderDataError)
        if not np.any(hdr['pixdim'][1:4] < 0):
            return ret
        ret.problem_msg = 'pixdim[1,2,3] should be positive'
        if fix:
            hdr['pixdim'][1:4] = np.abs(hdr['pixdim'][1:4])
            ret.fix_msg = 'setting to abs of pixdim values'
            return ret
        ret.problem_level = 40
        return ret

'''        

from StringIO import StringIO

import logging

from nibabel.testing import assert_true, assert_false, \
     assert_equal, assert_not_equal, assert_raises

from nibabel.batteryrunners import BatteryRunner, Report

# define some trivial functions as checks
def chk1(obj, fix=True):
    ret = Report(obj, KeyError)
    if obj.has_key('testkey'):
        return ret
    ret.problem_msg = 'no "testkey"'
    if fix:
        obj['testkey'] = 1
        ret.fix_msg = 'added "testkey"'
    else:
        ret.problem_level = 20
    return ret

def chk2(obj, fix=True):
    # Can return different codes for different errors in same check
    ret = Report(obj)
    try:
        ok = obj['testkey'] == 0
    except KeyError:
        ret.problem_msg = 'no "testkey"'
        ret.error = KeyError
        if fix:
            obj['testkey'] = 1
            ret.fix_msg = 'added "testkey"'
        else:
            ret.problem_level = 20
        return ret
    if ok:
        return ret
    ret.problem_msg = '"testkey" != 0'
    ret.error = ValueError
    if fix:
        ret.fix_msg = 'set "testkey" to 0'
        obj['testkey'] = 0
    else:
        ret.problem_level = 10
    return ret

def chk_warn(obj, fix=True):
    ret = Report(obj, KeyError)
    if obj.has_key('anotherkey'):
        return ret
    ret.problem_msg = 'no "anotherkey"'
    if fix:
        obj['anotherkey'] = 'a string'
        ret.fix_msg = 'added "anotherkey"'
    else:
        ret.problem_level = 30
    return ret

def chk_error(obj, fix=True):
    ret = Report(obj, KeyError)
    if obj.has_key('thirdkey'):
        return ret
    ret.problem_msg = 'no "thirdkey"'
    if fix:
        obj['anotherkey'] = 'a string'
        ret.fix_msg = 'added "anotherkey"'
    else:
        ret.problem_level = 40
    return ret


def test_init_basic():
    # With no args, raise
    yield assert_raises, TypeError, BatteryRunner
    # Len returns number of checks
    battrun = BatteryRunner((chk1,))
    yield assert_equal, len(battrun), 1
    battrun = BatteryRunner((chk1,chk2))
    yield assert_equal, len(battrun), 2


def test_report_strings():
    rep = Report()
    yield assert_not_equal, rep.__str__(), ''
    yield assert_equal, rep.message, ''
    str_io = StringIO()
    rep.write_raise(str_io)
    yield assert_equal, str_io.getvalue(), ''
    rep = Report('', ValueError, 20, 'msg', 'fix')
    rep.write_raise(str_io)
    yield assert_equal, str_io.getvalue(), ''
    rep.problem_level = 30
    rep.write_raise(str_io)
    yield assert_equal, str_io.getvalue(), 'Level 30: msg; fix\n'
    str_io.truncate(0)
    # No fix string, no fix message
    rep.fix_msg = ''
    rep.write_raise(str_io)
    yield assert_equal, str_io.getvalue(), 'Level 30: msg\n'
    rep.fix_msg = 'fix'
    str_io.truncate(0)
    # If we drop the level, nothing goes to the log
    rep.problem_level = 20
    rep.write_raise(str_io)
    yield assert_equal, str_io.getvalue(), ''
    # Unless we set the default log level in the call
    rep.write_raise(str_io, log_level=20)
    yield assert_equal, str_io.getvalue(), 'Level 20: msg; fix\n'
    str_io.truncate(0)
    # If we set the error level down this low, we raise an error
    yield assert_raises, ValueError, rep.write_raise, str_io, 20
    # But the log level wasn't low enough to do a log entry
    yield assert_equal, str_io.getvalue(), ''
    # Error still raised with lower log threshold, but now we do get a
    # log entry
    yield assert_raises, ValueError, rep.write_raise, str_io, 20, 20
    yield assert_equal, str_io.getvalue(), 'Level 20: msg; fix\n'
    # If there's no error, we can't raise
    str_io.truncate(0)
    rep.error = None
    rep.write_raise(str_io, 20)
    yield assert_equal, str_io.getvalue(), ''


def test_logging():
    rep = Report('', ValueError, 20, 'msg', 'fix')
    str_io = StringIO()
    logger = logging.getLogger('test.logger')
    logger.setLevel(30) # defaultish level
    logger.addHandler(logging.StreamHandler(str_io))
    rep.log_raise(logger)
    yield assert_equal, str_io.getvalue(), ''
    rep.problem_level = 30
    rep.log_raise(logger)
    yield assert_equal, str_io.getvalue(), 'msg; fix\n'
    str_io.truncate(0)
    
    
def test_checks():
    battrun = BatteryRunner((chk1,))
    reports = battrun.check_only({})
    yield assert_equal, reports[0], Report({},
                                           KeyError,
                                           20,
                                           'no "testkey"',
                                           '')
    obj, reports = battrun.check_fix({})    
    yield assert_equal, reports[0], Report({'testkey':1},
                                           KeyError,
                                           0,
                                           'no "testkey"',
                                           'added "testkey"')
    yield assert_equal, obj, {'testkey':1}
    battrun = BatteryRunner((chk1,chk2))
    reports = battrun.check_only({})
    yield assert_equal, reports[0], Report({},
                                           KeyError,
                                           20,
                                           'no "testkey"',
                                           '')
    yield assert_equal, reports[1], Report({},
                                           KeyError,
                                           20,
                                           'no "testkey"',
                                           '')
    obj, reports = battrun.check_fix({})
    # In the case of fix, the previous fix exposes a different error
    # Note, because obj is mutable, first and second point to modified
    # (and final) dictionary
    output_obj = {'testkey':0}
    yield assert_equal, reports[0], Report(output_obj,
                                           KeyError,
                                           0,
                                           'no "testkey"',
                                           'added "testkey"')
    yield assert_equal, reports[1], Report(output_obj,
                                           ValueError,
                                           0,
                                           '"testkey" != 0',
                                           'set "testkey" to 0')
    yield assert_equal, obj, output_obj
