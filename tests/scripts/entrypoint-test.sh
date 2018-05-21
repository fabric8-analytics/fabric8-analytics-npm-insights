#!/usr/bin/env bash

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
echo "*****************************************"

pytest /tests/unit_tests/
