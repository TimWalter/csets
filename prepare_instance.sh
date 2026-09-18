#!/bin/bash

# prepare_instance.sh — untimed setup before each instance (CORA-COMP interface).
# Arguments (the interface version, then the instance's instances.csv columns in file order):
# - $1: interface version string, e.g. "v1"
# - $2: benchmark, e.g. "zonotope" or "zonotope-batched"
# - $3: instance,  e.g. "matMul-500d-b10-gpu"
# - $4: params,    JSON object with everything the tool needs
#
# csets needs no per-instance setup: input generation, JIT compilation and the solve all
# belong to the measured run_instance.sh. A nonzero exit code would skip the instance.

set -e

VERSION_STRING="v1"
if [ "$1" != "$VERSION_STRING" ]; then
    echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
    exit 1
fi

exit 0
