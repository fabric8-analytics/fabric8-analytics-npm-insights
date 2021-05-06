#!/bin/bash -ex

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

pushd "${SCRIPT_DIR}/.." > /dev/null

# test coverage threshold
COVERAGE_THRESHOLD=80

export TERM=xterm
TERM=${TERM:-xterm}

# set up terminal colors
NORMAL=$(tput sgr0)
RED=$(tput bold && tput setaf 1)
GREEN=$(tput bold && tput setaf 2)
YELLOW=$(tput bold && tput setaf 3)

check_python_version() {
    python3 tools/check_python_version.py 3 6
}

check_python_version

PYTHONPATH=$(pwd)
export PYTHONPATH

echo "Create Virtualenv for Python deps ..."
function prepare_venv() {
    VIRTUALENV=$(which virtualenv)
    if [ $? -eq 1 ]
    then
        # python36 which is in CentOS does not have virtualenv binary
        VIRTUALENV=$(which virtualenv-3)
    fi

    ${VIRTUALENV} -p python3 venv && source venv/bin/activate
    if [ $? -ne 0 ]
    then
        printf "%sPython virtual environment can't be initialized%s" "${RED}" "${NORMAL}"
        exit 1
    fi
    printf "%sPython virtual environment initialized%s\n" "${YELLOW}" "${NORMAL}"
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
    pip3 install -r tests/requirements.txt
}

[ "$NOVENV" == "1" ] || prepare_venv || exit 1

locale charmap
set -ex

export RADONFILESENCODING=UTF-8
echo "*****************************************"
echo "*** Cyclomatic complexity measurement ***"
echo "*****************************************"
radon cc -s -a -i venv .

echo "*****************************************"
echo "*** Maintainability Index measurement ***"
echo "*****************************************"
radon mi -s -i venv .

echo "*****************************************"
echo "*** Unit tests ***"
echo "*****************************************"
PYTHONDONTWRITEBYTECODE=1 python3 "$(which pytest)" --cov=recommendation_engine/ --cov-report=xml --cov-fail-under=$COVERAGE_THRESHOLD -vv tests/unit_tests/
printf "%stests passed%s\n\n" "${GREEN}" "${NORMAL}"

popd > /dev/null