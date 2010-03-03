''' Battery runner classes and Report classes

These classes / objects are for generic checking / fixing batteries

The class will run a series of checks.

The checks may run fixes on the passed object, but if so the object has
to be mutable.

See below for what ``checks`` are.

To run checks only, and return problem report objects::

   btrun = BatteryRunner(checks)
   report_seq =  btrun.check_only(obj)

To run checks and fixes, returning fixed object, problem report object,
with possible fix messages::

   fixed_obj, report_seq = btrun.check_fix(obj)

Reports are iterable things, where the elements in the iterations are
``Problems``, with attributes ``obj``, ``error``, ``problem_level``,
``problem_msg``, and possibly empty ``fix_msg``.  The ``problem_level``
is an integer, giving the level of problem, from 0 (no problem) to 50
(very bad problem).  The levels follow the log levels from the logging
module.  The ``error`` can be one of ``None`` if no error to suggest, or
an Exception class that the user might consider raising for this
sitation.  The ``problem_msg`` and ``fix_msg`` are human readable
strings that should explain what happened.

The checks are a sequence of callables, looking like this::

   report = chk(obj, fix=False)

or::

   report_seq = chk(obj, fix=True)

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
            else:
                return ret
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
        ret.problem_level = 10
        ret.problem_msg = 'bitpix does not match datatype')
        if not fix:
            return ret
        hdr['bitpix'] = bitpix # inplace modification
        ret.fix_msg = 'setting bitpix to match datatype'
        return ret

    # or the pixdims

    def chk_pixdims(hdr, fix=True):
        ret = Report(hdr, HeaderDataError)
        if not np.any(hdr['pixdim'][1:4] < 0):
            return ret
        ret.problem_level = 40
        ret.problem_msg = 'pixdim[1,2,3] should be positive'
        if fix:
            hdr['pixdim'][1:4] = np.abs(hdr['pixdim'][1:4])
            ret.fix_msg = 'setting to abs of pixdim values'
        return ret

'''

class BatteryRunner(object):

    def __init__(self, checks):
        self._checks = checks

    def check_only(self, obj):
        reports = []
        for check in self._checks:
            reports.append(check(obj, False))
        return reports

    def check_fix(self, obj):
        reports = []
        for check in self._checks:
            report = check(obj, True)
            obj = report.obj
            reports.append(report)
        return obj, reports

    def __len__(self):
        return len(self._checks)


class Report(object):
    def __init__(self,
                 obj=None,
                 error=None,
                 problem_level=0,
                 problem_msg='',
                 fix_msg=''):
        ''' Initialize report with values

        Parameters
        ----------
        obj : object
           object tested, possibly fixed.  Default is None
        error : None or Exception
           Error to raise if raising error for this check.  If None,
           no error can be raised for this check (it was probably
           normal).
        problem_level : int
           level of problem.  From 0 (no problem) to 50 (severe
           problem).  If the report originates from a fix, then this
           is the level of the problem remaining after the fix.
           Default is 0
        problem_msg : string
           String describing problem detected. Default is ''
        fix_msg : string
           String describing any fix applied.  Default is ''.

        Examples
        --------
        >>> rep = Report()
        >>> rep.problem_level
        0
        >>> rep = Report((), TypeError, 10)
        >>> rep.problem_level
        10

        '''
        self.obj = obj
        self.error = error
        self.problem_level = problem_level
        self.problem_msg = problem_msg
        self.fix_msg = fix_msg

    def __eq__(self, other):
        ''' Test for equality

        Parameters
        ----------
        other : object
           report-like object to test equality

        Examples
        --------
        >>> rep = Report(problem_level=10)
        >>> rep2 = Report(problem_level=10)
        >>> rep == rep2
        True
        >>> rep3 = Report(problem_level=20)
        >>> rep == rep3
        False
        '''
        return (self.__dict__ == other.__dict__)

    def __ne__(self, other):
        ''' Test for equality

        Parameters
        ----------
        other : object
           report-like object to test equality

        See __eq__ docstring for examples
        '''
        return (self.__dict__ != other.__dict__)

    def __str__(self):
        ''' Printable string for object '''
        return self.__dict__.__str__()

    @property
    def message(self):
        ''' formatted message string, including fix message if present
        '''
        if self.fix_msg:
            return '; '.join((self.problem_msg, self.fix_msg))
        return self.problem_msg

    def log_raise(self, logger, error_level=40):
        ''' Log problem, raise error if problem >= `error_level`

        Parameters
        ----------
        logger : log
           log object, implementing ``log`` method
        error_level : int, optional
           If ``self.problem_level`` >= `error_level`, raise error
        '''
        logger.log(self.problem_level, self.message)
        if self.problem_level and self.problem_level >= error_level:
            if self.error:
                raise self.error(self.problem_msg)

    def write_raise(self, stream, error_level=40, log_level=30):
        if self.problem_level >= log_level:
            stream.write('Level %s: %s\n' %
                         (self.problem_level, self.message))
        if self.problem_level and self.problem_level >= error_level:
            if self.error:
                raise self.error(self.problem_msg)
