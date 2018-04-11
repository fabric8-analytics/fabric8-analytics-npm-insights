#!/usr/bin/env bash

TEST_IMAGE_NAME='chester-tests'

gc() {
    docker rmi -f `make get-image-name`
    docker rmi -f ${TEST_IMAGE_NAME}
}

if [[ "$CI" -eq "0" ]];
then
    make docker-build-test
    docker run ${TEST_IMAGE_NAME}
    docker stop ${TEST_IMAGE_NAME}
    trap gc EXIT SIGINT
else
    # CI instance will be torn down anyway, don't need to waste time on gc
    docker run ${TEST_IMAGE_NAME}
fi
