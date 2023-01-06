#!/bin/bash

echo Running tests

source tools/ci/activate.sh

set -eu

# Required variables
echo CHECK_TYPE = $CHECK_TYPE

set -x

export NIBABEL_DATA_DIR="$PWD/nibabel-data"

if [ "${CHECK_TYPE}" == "style" ]; then
    # Run styles only on core nibabel code.
    flake8 nibabel
elif [ "${CHECK_TYPE}" == "doctest" ]; then
    make -C doc html && make -C doc doctest
elif [ "${CHECK_TYPE}" == "test" ]; then
    # Change into an innocuous directory and find tests from installation
    mkdir for_testing
    cd for_testing
    cp ../.coveragerc .
    pytest --doctest-modules --doctest-plus --cov nibabel --cov-report xml \
        --junitxml=test-results.xml -v --pyargs nibabel -n auto
elif [ "${CHECK_TYPE}" == "typing" ]; then
    mypy nibabel
else
    false
fi

set +eux

echo Done running tests
