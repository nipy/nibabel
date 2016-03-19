#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Wrapper to run setup.py using setuptools."""

import setuptools  # flake8: noqa ; needed to monkeypatch dist_utils

###############################################################################
# Call the setup.py script, injecting the setuptools-specific arguments.

if __name__ == '__main__':
    exec(open('setup.py', 'rt').read(), dict(__name__='__main__'))
