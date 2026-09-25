"""
Break one CORA-COMP instance's wall-clock time down into where it actually goes.

    python benchmark/profile_instance.py zonotope/matMul-1000d-gpu
    python benchmark/profile_instance.py zonotope-batched/contains-10d-b100-cpu --log-compiles
    python benchmark/profile_instance.py '<params json>' --trace /tmp/trace   # Perfetto/TensorBoard trace

Runs the instance through the same `cora_comp.Instance` the daemon uses, timing each phase:
interpreter start, the jax / moreau / csets imports, backend (CUDA) initialisation, solver
construction and XLA compilation (all of which the daemon does in the untimed
prepare_instance.sh), then the measured part: input generation and the repeated operation.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CATALOG = REPO / "benchmark" / ".cache" / "instances.csv"


def process_age() -> float:
    """Seconds since this process was spawned (Linux, 10 ms resolution)."""
    try:
        start_ticks = int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19])
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        return uptime - start_ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return float("nan")


T_SPAWN_AGE = process_age()


def lookup(spec: str, catalog: Path) -> dict:
    if spec.lstrip().startswith("{"):
        return json.loads(spec)
    bench, _, inst = spec.rpartition("/")
    bench = bench or "zonotope"
    if not catalog.exists():
        sys.exit(f"{catalog} not found; run `benchmark/local_eval.py run --mode first` once, or pass params JSON.")
    with open(catalog, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row["benchmark"] == bench and row["instance"] == inst:
                return json.loads(row["params"])
    sys.exit(f"No instance {bench}/{inst} in {catalog}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("instance", help="'<benchmark>/<instance>' (benchmark defaults to zonotope) or params JSON")
    p.add_argument("--catalog", type=Path, default=CATALOG)
    p.add_argument("--reps", type=int, help="override params.repetition")
    p.add_argument("--log-compiles", action="store_true", help="print every XLA compilation")
    p.add_argument("--trace", help="write a jax.profiler trace of the steady-state loop to this directory")
    args = p.parse_args()

    params = lookup(args.instance, args.catalog)
    if args.reps:
        params["repetition"] = args.reps
    # Same environment as run_instance.sh, set before jax is imported.
    os.environ["JAX_PLATFORMS"] = "cuda,cpu" if params["device"] == "gpu" else "cpu"
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "benchmark"))
    import warnings
    warnings.filterwarnings("ignore")

    phases = [("interpreter start (spawn -> first line)", T_SPAWN_AGE)]
    mark = [time.perf_counter()]

    def phase(name):
        now = time.perf_counter()
        phases.append((name, now - mark[0]))
        mark[0] = now

    phase("argument parsing")
    import jax
    phase("import jax")
    import moreau.jax  # noqa: F401
    phase("import moreau.jax")
    import csets  # noqa: F401
    phase("import csets (beartype/jaxtyping import hook)")
    import cora_comp
    phase("import cora_comp")

    if args.log_compiles:
        jax.config.update("jax_log_compiles", True)
    device = cora_comp.configure(params)
    if device is None:
        sys.exit(f"No {params['device']} device available")
    jax.block_until_ready(jax.device_put(jax.numpy.zeros(1), device) + 1)
    phase(f"backend init ({device.platform})")

    instance = cora_comp.Instance(params)
    phase(f"Instance(): programs + Moreau solver (shapes only, {instance.shards} shard(s))")
    # As Instance.compile, but timed per program: one generate and one operation per shard.
    for k, shard_device in enumerate(instance.devices):
        with jax.default_device(shard_device):
            instance.generate[k] = instance.generate[k].lower().compile()
    phase("compile generate (shapes only)")
    for k, shard_device in enumerate(instance.devices):
        with jax.default_device(shard_device):
            instance.operation[k] = instance.operation[k].lower(0, instance.generate[k].out_info).compile()
    phase("compile operation (shapes only)")
    daemon_start = len(phases)  # everything above runs in prepare_instance.sh with the daemon

    reps = params["repetition"]
    from contextlib import nullcontext
    with jax.profiler.trace(args.trace) if args.trace else nullcontext():
        time_generate, time_operation, output = instance.run()  # the measured part, as the daemon runs it
    phases += [("[timed] generate inputs", time_generate), (f"[timed] operation x{reps}", time_operation)]

    total = sum(dt for _, dt in phases if dt == dt)  # a NaN spawn age (off Linux) is skipped
    timed = sum(dt for _, dt in phases[daemon_start:])
    name = args.instance if not args.instance.lstrip().startswith("{") else params["operation"]
    print(f"\n{name}   device={device}   x64={jax.config.jax_enable_x64}   jax {jax.__version__}")
    print(f"params: {json.dumps(params)}\n")
    for k, (label, dt) in enumerate(phases):
        if k == daemon_start:
            print(f"  {'-- measured with the daemon; above is prepare_instance.sh --':48s}")
        bar = "#" * int(round(40 * dt / total)) if dt == dt else ""
        print(f"  {label:48s} {dt * 1e3:10.1f} ms  {100 * dt / total:5.1f}%  {bar}")
    print(f"\n  measured, with the daemon (+ a few ms socket):   {timed * 1e3:10.1f} ms")
    print(f"  measured, without it (fresh process, lazy jit): ~{total * 1e3:9.1f} ms")
    print(f"  per repetition: {phases[-1][1] / reps * 1e3:.3f} ms")
    if params["operation"] == "contains":
        print(f"  contains: {int(output.sum())}/{output.size} true")
    if args.trace:
        print(f"  trace written to {args.trace} (open with Perfetto / TensorBoard)")

if __name__ == "__main__":
    main()
