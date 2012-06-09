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
from functools import partial
import urllib
import cgi

import jinja2

from nibabel import dft

# this is the directory containing the DICOM data, or None for all cached data
BASE_DIR = '/path/to/DICOM'
BASE_DIR = None

# default setting for whether to follow symlinks in BASE_DIR. Python 2.5 only
# accepts False for this setting, Python >= 2.6 accepts True or False
FOLLOWLINKS=False

# Define routine to get studies
studies_getter = partial(dft.get_studies, followlinks=FOLLOWLINKS)


def html_unicode(u):
    return cgi.escape(u.encode('utf-8'))


template_env = jinja2.Environment(autoescape=True)
template_env.filters['urlquote'] = urllib.quote

index_template = """<html><head><title>data</title></head>
<body>
Home
<br />
<br />
{% for p in patients|sort %}
    Patient: <a href="{{ p|urlquote }}/">{{ p }}</a>
    <br />
    {% if patients[p]|length == 1 %}
        1 study
    {% else %}
        {{ patients[p]|length }} studies
    {% endif %}
    <br />
{% endfor %}
</body>
</html>
"""

patient_template = """<html><head><title>data</title></head>
<body>
<a href="../">Home</a> -&gt; Patient {{ studies[0].patient_name_or_uid() }}
<br />
<br />
Patient name: {{ studies[0].patient_name }}
<br />
Patient ID: {{ studies[0].patient_id }}
<br />
Patient birth date: {{ studies[0].patient_birth_date }}
<br />
Patient sex: {{ studies[0].patient_sex }}
<br />
<ul>
{% for s in studies %}
    <li><a href="{{ s.date|urlquote }}_{{ s.time|urlquote }}/">Study {{ s.uid }}</a></li>
    <ul>
    <li>Date: {{ s.date }}</li>
    <li>Time: {{ s.time }}</li>
    <li>Comments: {{ s.comments }}</li>
    <li>Series: {{ s.series|length }}</li>
{% endfor %}
</ul>
</body>
</html>
"""

patient_date_time_template = """
<html><head><title>data</title></head>
<body>
<a href="../../">Home</a> -&gt; <a href="../../{{ study.patient_name_or_uid() }}/">Patient {{ study.patient_name_or_uid() }}</a> -&gt; Study {{ study.date}} {{ study.time }}
<br />
<br />
Patient name: <a href="../../{{ study.patient_name_or_uid() }}/">{{ study.patient_name }}</a>
<br />
Study UID: {{ study.uid }}
<br />
Study date: {{ study.date }}
<br />
Study time: {{ study.time }}
<br />
Study comments: {{ study.comments }}
{% if study.series|length == 0 %}
    <br />
    No series.
{% else %}
    <ul>
    {% for s in study.series %}
        <li>Series {{ s.number }} (<a href="{{ s.number }}/nifti">NIfTI</a>)</li>
        <ul>
        <li>Series UID: {{ s.uid }}</li>
        <li>Series description: {{ s.description }}</li>
        <li>Series dimensions: {{ s.rows }}x{{ s.columns }}x{{ s.storage_instances|length }}</li>
        </ul>
        <img src="{{ s.number }}/png" />
    {% endfor %}
    </ul>
{% endif %}
</body>
</html>
"""

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

def index(environ):
    patients = {}
    for s in studies_getter(BASE_DIR):
        patients.setdefault(s.patient_name_or_uid(), []).append(s)
    template = template_env.from_string(index_template)
    return template.render(patients=patients).encode('utf-8')

def patient(patient):
    studies = [ s for s in studies_getter() if s.patient_name_or_uid() == patient ]
    if len(studies) == 0:
        raise HandlerError('404 Not Found', 'patient %s not found\n' % patient)
    studies.sort(study_cmp)
    template = template_env.from_string(patient_template)
    return template.render(studies=studies).encode('utf-8')

def patient_date_time(patient, date_time):
    study = None
    for s in studies_getter():
        if s.patient_name_or_uid() != patient:
            continue
        if date_time != '%s_%s' % (s.date, s.time):
            continue
        study = s
        break
    if study is None:
        raise HandlerError, ('404 Not Found', 'study not found')
    template = template_env.from_string(patient_date_time_template)
    return template.render(study=study).encode('utf-8')

def nifti(patient, date_time, scan):
    study = None
    for s in studies_getter():
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
    for s in studies_getter():
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
