#!/usr/bin/env bash

set -e

# test coverage threshold
COVERAGE_THRESHOLD=50

check_python_version() {
    python3 tools/check_python_version.py 3 6
}

check_python_version

radon --version

locale charmap

export RADONFILESENCODING=UTF-8

echo "*****************************************"
echo "*** Cyclomatic complexity measurement ***"
echo "*****************************************"
radon cc -s -a -i usr .

echo "*****************************************"
echo "*** Maintainability Index measurement ***"
echo "*****************************************"
radon mi -s -i usr .

echo "*****************************************"
echo "*** Unit tests ***"
echo "*the****************************************"

cd shared
pytest --cov=/recommendation_engine/ --cov-report=xml --cov-fail-under=$COVERAGE_THRESHOLD -vv /tests/unit_tests/
