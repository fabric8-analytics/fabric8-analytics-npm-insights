#!/bin/bash

directories="recommendation_engine tests"

# checks for the whole directories
for directory in $directories
do
    grep -r -n "TODO: " "$directory"
done
