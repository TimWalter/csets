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
# Everything this script does is timed by the harness, so it only hands params to the warm
# daemon started by prepare_instance.sh and waits for the verdict; the daemon generates the
# inputs and repeats the operation (benchmark/cora_comp.py). Without a daemon, the instance
# runs in a fresh process instead, paying Python, imports and compilation inside the measurement.

set -u

VERSION_STRING="v1"
if [ "$1" != "$VERSION_STRING" ]; then
    echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
    exit 1
fi

case "$0" in */*) HERE="${0%/*}" ;; *) HERE=. ;; esac  # no subshell: this is measured
. "$HERE/benchmark/client.sh"

PARAMS="$4"
RESULTS_FILE="${@: -1}"

ask "run	$RESULTS_FILE	$PARAMS"
case $? in
    0)  # The daemon's log only matters when the instance did not finish.
        [ "$REPLY" = finished ] || cat "$SRV_DIR/job.log"
        exit 0 ;;
    1)  echo "[run] no csets daemon; running directly" ;;
    *)  echo "[run] the csets daemon died during the instance"
        cat "$SRV_DIR/job.log"
        printf 'result\nerror\n' > "$RESULTS_FILE"
        exit 0 ;;
esac

# Fallback: a fresh process. cpu instances must never touch the GPU, so JAX only gets the CPU
# backend; gpu instances keep the CPU backend too, since Moreau's JAX bindings need it.
cd "$HERE"
PYTHON=.venv/bin/python
case "$PARAMS" in
    *'"device": "gpu"'*|*'"device":"gpu"'*) export JAX_PLATFORMS=cuda,cpu ;;
    *) export JAX_PLATFORMS=cpu ;;
esac
HERE=. && export XLA_FLAGS="$(xla_flags)"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS=ignore

rm -f "$RESULTS_FILE"
"$PYTHON" benchmark/cora_comp.py "$PARAMS" "$RESULTS_FILE"
STATUS=$?
if [ $STATUS -ne 0 ] || [ ! -s "$RESULTS_FILE" ]; then
    echo "Driver exited with status $STATUS; reporting error."
    printf 'result\nerror\n' > "$RESULTS_FILE"
fi
exit 0
