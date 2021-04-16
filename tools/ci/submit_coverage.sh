#!/bin/bash

echo Submitting coverage

source tools/ci/activate.sh

set -eu

set -x

COVERAGE_FILE="for_testing/coverage.xml"

if [ -e "$COVERAGE_FILE" ]; then
    # Pin codecov version to reduce scope for malicious updates
    python -m pip install "codecov==2.1.11"
    python -m codecov --file for_testing/coverage.xml
fi

set +eux

echo Done submitting coverage
