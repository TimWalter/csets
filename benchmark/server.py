"""
The warm csets daemon for CORA-COMP.

Starting Python, importing JAX/Moreau/csets, creating the CUDA context and compiling every
program takes far longer than most instances. prepare_instance.sh (untimed) therefore starts
this daemon once and asks it to compile each instance from shapes alone; run_instance.sh then
only sends the instance over a localhost socket and waits for the verdict, using bash's
`/dev/tcp`, so no process is spawned inside the measurement. This is the same split the other
JAX entry (CORA.jax) uses.

    python benchmark/server.py <server-dir>

One request per connection, one line each way:

    ping                                     -> pong
    warm <TAB> params JSON                   -> compiled | unsupported | warm failed
    run <TAB> results file <TAB> params JSON -> the verdict

Requests are served one at a time, so a daemon still busy with an instance the harness gave up
on does not answer `ping`, and prepare_instance.sh replaces it.
"""
import contextlib
import gc
import json
import os
import socket
import sys
import traceback
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402

import cora_comp  # noqa: E402

OPERATIONS = ("startup", "generateRandom", "randPoint", "supportFunc", "matMul", "minkSum", "contains")
KEEP = 2  # prepared instances held at once; each is run once, right after its prepare

_prepared: "OrderedDict[str, cora_comp.Instance]" = OrderedDict()


def port() -> int:
    return int(os.environ.get("CSETS_PORT", "47931"))


def prepare(params: str) -> cora_comp.Instance:
    """The compiled instance for these params, from the cache or built now."""
    instance = _prepared.pop(params, None)
    if instance is None:
        instance = cora_comp.Instance(json.loads(params))
        instance.compile()
    _prepared[params] = instance
    while len(_prepared) > KEEP:
        _prepared.popitem(last=False)
    return instance


def warm_up() -> None:
    """Run every operation once, small, on every device, so the one-off costs of the process
    (first traces, the type-checking hook, Moreau's setup, CUDA kernels) are paid here."""
    devices = ["cpu"] + (["gpu"] if cora_comp.configure({"device": "gpu"}) is not None else [])
    for device in devices:
        for batch in ({}, {"batch_size": 2}):
            for operation in OPERATIONS:
                params = {"set": "zonotope", "operation": operation, "dim": 2, "generators": 4,
                          "device": device, "repetition": 1, "points": 2, "type": "standard", **batch}
                instance = cora_comp.Instance(params)
                instance.compile()
                with contextlib.redirect_stdout(open(os.devnull, "w")):
                    cora_comp.run_instance(instance, os.devnull)
    print(f"[csets] warmed up on {', '.join(devices)}; {len(jax.devices('cpu'))} XLA CPU device(s), "
          f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}", flush=True)


def handle(request: str, log_path: str) -> str:
    kind, _, rest = request.partition("\t")
    if kind == "ping":
        return "pong"
    if kind == "warm":
        try:
            return "unsupported" if prepare(rest).unsupported else "compiled"
        except Exception:
            traceback.print_exc()
            return "warm failed"
    if kind != "run":
        return f"unknown request {kind!r}"
    results_file, _, params = rest.partition("\t")
    with open(log_path, "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            instance = prepare(params)  # compiles here, inside the measurement, if prepare did not
            _prepared.pop(params, None)  # run once; dropped so its solver and programs can be freed
            return cora_comp.run_instance(instance, results_file)
        except Exception:
            traceback.print_exc()
            cora_comp.write_result(results_file, cora_comp.ERROR)
            return cora_comp.ERROR


def serve(srv_dir: str) -> None:
    """Warm up, then serve until killed; the pid file is how prepare_instance.sh finds this process."""
    Path(srv_dir, "server.pid").write_text(str(os.getpid()))
    warm_up()
    log_path = os.path.join(srv_dir, "job.log")
    with socket.create_server(("127.0.0.1", port())) as srv:
        print(f"[csets] serving on 127.0.0.1:{port()}", flush=True)
        while True:
            conn, _ = srv.accept()
            with conn, conn.makefile("r", encoding="utf-8") as reader:
                request = reader.readline().rstrip("\n")
                reply = handle(request, log_path)
                with contextlib.suppress(OSError):
                    conn.sendall((reply + "\n").encode())
            # Free the finished instance's buffers. Only after a run: the next request is then an
            # untimed prepare, while a run follows its prepare at once and would wait for this.
            if request.startswith("run"):
                gc.collect()


if __name__ == "__main__":
    serve(sys.argv[1])
