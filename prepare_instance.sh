#!/bin/bash

# prepare_instance.sh — untimed setup before each instance (CORA-COMP interface).
# Arguments (the interface version, then the instance's instances.csv columns in file order):
# - $1: interface version string, e.g. "v1"
# - $2: benchmark, e.g. "zonotope" or "zonotope-batched"
# - $3: instance,  e.g. "matMul-500d-b10-gpu"
# - $4: params,    JSON object with everything the tool needs
#
# Makes sure the warm csets daemon (benchmark/server.py) is up, so that Python startup, the
# jax/moreau/csets imports and CUDA initialisation never land in a measurement, and asks it
# to compile this instance's programs from their shapes. No inputs are generated here; that
# stays in run_instance.sh. A daemon that does not answer — dead, or still busy with an
# instance the harness timed out — is replaced.
#
# Always exits 0: without a daemon, run_instance.sh runs the instance directly.

set -u

VERSION_STRING="v1"
if [ "$1" != "$VERSION_STRING" ]; then
    echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
    exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/benchmark/client.sh"
START_TIMEOUT="${CSETS_START_TIMEOUT:-600}"

warm() {
    # Most of the harness's 600 s for a prepare: what compiles here is not compiled in the measurement.
    ask "warm	$4" "${CSETS_WARM_TIMEOUT:-540}" && echo "[prepare] $3: $REPLY"
}

if ask ping 5 && [ "$REPLY" = pong ]; then
    warm "$@"
    exit 0
fi

mkdir -p "$SRV_DIR"
PID="$(cat "$SRV_DIR/server.pid" 2>/dev/null)"
rm -f "$SRV_DIR/server.pid"
# The pid may have been reused since, so only a process that still is the daemon is killed.
if [ -n "$PID" ] && grep -qa "benchmark/server.py" "/proc/$PID/cmdline" 2>/dev/null; then
    echo "[prepare] replacing the unresponsive csets daemon"
    kill -KILL "$PID"
    for _ in $(seq 50); do ask ping 1; [ $? -eq 1 ] && break; sleep 0.1; done
fi

FLAGS="$(xla_flags)"
echo "[prepare] starting the csets daemon (XLA_FLAGS='$FLAGS')"
# Own session, so the harness's process-group kill of a timed-out run leaves it alone, and no
# inherited stdout, which the harness waits on. JAX gets every platform: the daemon serves cpu
# and gpu instances, and Moreau's JAX bindings need the CPU backend in either case.
( cd "$HERE" && unset JAX_PLATFORMS && XLA_FLAGS="$FLAGS" PYTHONWARNINGS=ignore exec setsid "$PYTHON" benchmark/server.py "$SRV_DIR" ) \
    > "$SRV_DIR/server.log" 2>&1 < /dev/null &

SECONDS=0
while [ "$SECONDS" -lt "$START_TIMEOUT" ]; do
    if ask ping 5 && [ "$REPLY" = pong ]; then
        echo "[prepare] csets daemon is up after ${SECONDS}s"
        warm "$@"
        exit 0
    fi
    kill -0 $! 2>/dev/null || break
    sleep 0.2
done

echo "[prepare] the csets daemon did not come up; run_instance.sh will run directly"
cat "$SRV_DIR/server.log"
exit 0
