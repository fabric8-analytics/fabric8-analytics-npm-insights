#!/usr/bin/env bash

set -e

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

pushd "${SCRIPT_DIR}/.." > /dev/null

TEST_IMAGE_NAME='chester-tests'

gc() {
    docker rmi -f $(make get-image-name)
    docker rmi -f "${TEST_IMAGE_NAME}"
}

if [[ "$CI" -eq "0" ]];
then
    make docker-build-test
    docker run -v "$PWD:/shared:rw,Z" ${TEST_IMAGE_NAME}
    trap gc EXIT SIGINT
else
    # CI instance will be torn down anyway, don't need to waste time on gc
    docker run -v "$PWD:/shared:rw,Z" ${TEST_IMAGE_NAME}
fi

popd > /dev/null
