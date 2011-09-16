# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
# Copyright (C) 2011 Christian Haselgrove

import sys
import traceback
import urllib
import cgi
from nibabel import dft

# this is the directory containing the DICOM data, or None for all cached data
base_dir = '/Users/ch/Desktop/umms/dft/trunk/data/t'
base_dir = None

class HandlerError:

    def __init__(self, status, output):
        self.status = status
        self.output = output
        return

def application(environ, start_response):
    try:
        (status, c_type, output) = handler(environ)
    except HandlerError, exc:
        status = exc.status
        output = exc.output
        c_type = 'text/plain'
    except:
        (exc_type, exc_value, exc_traceback) = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        status = '500 Internal Server Error'
        output = ''.join(lines)
        c_type = 'text/plain'
    response_headers = [('Content-Type', c_type), 
                        ('Content-Length', str(len(output)))]
    if c_type == 'image/nifti':
        response_headers.append(('Content-Disposition', 'attachment; filename=image.nii'))
    start_response(status, response_headers)
    return [output]

def handler(environ):
    if environ['PATH_INFO'] == '' or environ['PATH_INFO'] == '/':
        return ('200 OK', 'text/html', index(environ))
    parts = environ['PATH_INFO'].strip('/').split('/')
    if len(parts) == 1:
        return ('200 OK', 'text/html', patient(parts[0]))
    if len(parts) == 2:
        return ('200 OK', 'text/html', patient_date_time(parts[0], parts[1]))
    if len(parts) == 4:
        if parts[3] == 'nifti':
            return ('200 OK', 'image/nifti', nifti(parts[0], parts[1], parts[2]))
        elif parts[3] == 'png':
            return ('200 OK', 'image/png', png(parts[0], parts[1], parts[2]))
    raise HandlerError('404 Not Found', "%s not found\n" % environ['PATH_INFO'])

def index(environ):
    patients = {}
    for s in dft.get_studies(base_dir):
        patients.setdefault(s.patient_name_or_uid(), []).append(s)
    output = ''
    output += '<html><head><title>data</title></head>\n'
    output += '<body>\n'
    output += 'Home\n'
    output += '<br />\n'
    output += '<br />\n'
    for p in sorted(patients):
        output += 'Patient: <a href="%s/">%s</a>\n' % (urllib.quote(p.encode('utf-8')), html_unicode(p))
        output += '<br />\n'
        if len(patients[p]) == 1:
            output += '1 study\n'
        else:
            output += '%d studies' % len(patients[p])
        output += '<br />\n'
    output += '</body>\n'
    output += '</html>\n'
    return output

def study_cmp(a, b):
    if a.date < b.date:
        return -1
    if a.date > b.date:
        return 1
    if a.time < b.time:
        return -1
    if a.time > b.time:
        return 1
    return 0

def html_unicode(u):
    return cgi.escape(u.encode('utf-8'))

def patient(patient):
    studies = [ s for s in dft.get_studies() if s.patient_name_or_uid() == patient ]
    if len(studies) == 0:
        raise HandlerError('404 Not Found', 'patient %s not found\n' % patient)
    studies.sort(study_cmp)
    output = ''
    output += '<html><head><title>data</title></head>\n'
    output += '<body>\n'
    output += '<a href="../">Home</a> -&gt; Patient %s\n' % html_unicode(studies[0].patient_name_or_uid())
    output += '<br />\n'
    output += '<br />\n'
    output += 'Patient name: %s\n' % html_unicode(studies[0].patient_name)
    output += '<br />\n'
    output += 'Patient ID: %s\n' % html_unicode(studies[0].patient_id)
    output += '<br />\n'
    output += 'Patient birth date: %s\n' % html_unicode(studies[0].patient_birth_date)
    output += '<br />\n'
    output += 'Patient sex: %s\n' % html_unicode(studies[0].patient_sex)
    output += '<br />\n'
    output += '<ul>\n'
    for s in studies:
        output += '<li><a href="%s_%s/">Study %s</a></li>\n' % (urllib.quote(s.date), urllib.quote(s.time), html_unicode(s.uid))
        output += '<ul>\n'
        output += '<li>Date: %s</li>\n' % html_unicode(s.date)
        output += '<li>Time: %s</li>\n' % html_unicode(s.time)
        output += '<li>Comments: %s</li>\n' % html_unicode(s.comments)
        output += '<li>Series: %d</li>\n' % len(s.series)
        output += '</ul>\n'
    output += '</body>\n'
    output += '</html>\n'
    return output

def patient_date_time(patient, date_time):
    study = None
    for s in dft.get_studies():
        if s.patient_name_or_uid() != patient:
            continue
        if date_time != '%s_%s' % (s.date, s.time):
            continue
        study = s
        break
    if study is None:
        raise HandlerError, ('404 Not Found', 'study not found')
    output = ''
    output += '<html><head><title>data</title></head>\n'
    output += '<body>\n'
    output += '<a href="../../">Home</a> -&gt; <a href="../../%s/">Patient %s</a> -&gt; Study %s %s\n' % (urllib.quote(study.patient_name_or_uid()), html_unicode(study.patient_name_or_uid()), html_unicode(study.date), html_unicode(study.time))
    output += '<br />\n'
    output += '<br />\n'
    output += 'Patient name: <a href="/../%s/">%s</a>\n' % (urllib.quote(study.patient_name_or_uid()), html_unicode(study.patient_name))
    output += '<br />\n'
    output += 'Study UID: %s\n' % html_unicode(study.uid)
    output += '<br />\n'
    output += 'Study date: %s\n' % html_unicode(study.date)
    output += '<br />\n'
    output += 'Study time: %s\n' % html_unicode(study.time)
    output += '<br />\n'
    output += 'Study comments: %s\n' % html_unicode(study.comments)
    if len(study.series) == 0:
        output += '<br />\n'
        output += 'No series.\n'
    else:
        output += '<ul>\n'
        for s in study.series:
            output += '<li>Series %s (<a href="%s/nifti">NIfTI</a>)</li>\n' % (html_unicode(s.number), html_unicode(s.number))
            output += '<ul>\n'
            output += '<li>Series UID: %s</li>\n' % html_unicode(s.uid)
            output += '<li>Series description: %s</li>\n' % html_unicode(s.description)
            output += '<li>Series dimensions: %dx%dx%d</li>\n' % (s.rows, s.columns, len(s.storage_instances))
            output += '</ul>\n'
            output += '<img src="%s/png" />\n' % html_unicode(s.number)
        output += '</ul>\n'
    output += '</body>\n'
    output += '</html>\n'
    return output

def nifti(patient, date_time, scan):
    study = None
    for s in dft.get_studies():
        if s.patient_name_or_uid() != patient:
            continue
        if date_time != '%s_%s' % (s.date, s.time):
            continue
        study = s
        break
    if study is None:
        raise HandlerError, ('404 Not Found', 'study not found')
    ser = None
    for series in s.series:
        if series.number != scan:
            continue
        ser = series
        break
    if ser is None:
        raise HandlerError, ('404 Not Found', 'series not found')
    return ser.as_nifti()

def png(patient, date_time, scan):
    study = None
    for s in dft.get_studies():
        if s.patient_name_or_uid() != patient:
            continue
        if date_time != '%s_%s' % (s.date, s.time):
            continue
        study = s
        break
    if study is None:
        raise HandlerError, ('404 Not Found', 'study not found')
    ser = None
    for series in s.series:
        if series.number != scan:
            continue
        ser = series
        break
    if ser is None:
        raise HandlerError, ('404 Not Found', 'series not found')
    index = len(ser.storage_instances) / 2
    return ser.as_png(index, True)

if __name__ == '__main__':
    import wsgiref.simple_server
    httpd = wsgiref.simple_server.make_server('', 8080, application)
    httpd.serve_forever()

# eof
