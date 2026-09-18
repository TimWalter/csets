#!/bin/bash

# run_instance.sh — run one measured instance with csets and report the verdict (CORA-COMP).
# Arguments (the interface version, then the instance's instances.csv columns in file order):
# - $1: interface version string, e.g. "v1"
# - $2: benchmark, e.g. "zonotope" or "zonotope-batched"
# - $3: instance,  e.g. "matMul-500d-b10-gpu"
# - $4: params,    JSON object with everything the tool needs, e.g. '{"set": "zonotope",
#                  "operation": "matMul", "dim": 500, "generators": 1000, "device": "gpu",
#                  "repetition": 100, "batch_size": 10}'
# A column added to the catalog later arrives as a further argument, in file order, and
# the results file to write is always the LAST argument.
#
# Everything this script does is timed by the harness. The work itself is in
# benchmark/cora_comp.py, which dispatches on params; this wrapper only selects the JAX
# backend and guarantees a verdict is written even if the driver crashes.

VERSION_STRING="v1"
if [ "$1" != "$VERSION_STRING" ]; then
    echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
    exit 1
fi

BENCHMARK="$2"
INSTANCE="$3"
PARAMS="$4"
RESULTS_FILE="${@: -1}"

cd "$(dirname "$0")"

DEVICE=$(printf '%s' "$PARAMS" | python3 -c 'import json,sys; print(json.load(sys.stdin)["device"])')
echo "Running $BENCHMARK/$INSTANCE on $DEVICE -> $RESULTS_FILE"

# cpu instances must never touch the GPU, so JAX only gets the CPU backend. gpu instances
# keep the CPU backend too: Moreau's JAX bindings stage through host callbacks that need it.
if [ "$DEVICE" = gpu ]; then
    export JAX_PLATFORMS=cuda,cpu
else
    export JAX_PLATFORMS=cpu
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS=ignore

rm -f "$RESULTS_FILE"
.venv/bin/python benchmark/cora_comp.py "$PARAMS" "$RESULTS_FILE"
STATUS=$?

if [ $STATUS -ne 0 ] || [ ! -s "$RESULTS_FILE" ]; then
    echo "Driver exited with status $STATUS; reporting error."
    printf 'result\nerror\n' > "$RESULTS_FILE"
fi
exit 0
