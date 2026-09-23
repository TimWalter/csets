#!/bin/bash
# client.sh — client side of the csets daemon (benchmark/server.py), sourced by
# prepare_instance.sh and run_instance.sh. Builtins only, since run_instance.sh calls it
# inside the measured region.
#
# Expects HERE (the repository root). Sets PYTHON, SRV_DIR, PORT; ask() sets REPLY.

PYTHON="$HERE/.venv/bin/python"
SRV_DIR="${CSETS_SERVER_DIR:-$HERE/.server}"
PORT="${CSETS_PORT:-47931}"
export CSETS_PORT="$PORT"

# ask REQUEST [TIMEOUT]: send one request line to the daemon and read its one-line reply
# into REPLY. Returns 1 if no daemon is listening, 2 if it did not answer.
ask() {
    REPLY=""
    { exec 3<>"/dev/tcp/127.0.0.1/$PORT"; } 2>/dev/null || return 1
    printf '%s\n' "$1" >&3
    if ! read -r ${2:+-t "$2"} REPLY <&3 2>/dev/null; then
        exec 3>&-
        return 2
    fi
    exec 3>&-
    return 0
}
