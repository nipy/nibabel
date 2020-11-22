#!/bin/bash

echo Uploading coverage to codecov

set -eux

pip install codecov
codecov

set +eux

echo Done uploading coverage to codecov

