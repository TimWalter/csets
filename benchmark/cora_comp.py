"""
CORA-COMP driver: one benchmark instance with csets, split into what may happen before the
measurement and what has to happen inside it.

- `Instance(params)` and `Instance.compile()` are shape-only setup: they pin the device, build
  the Moreau solver (its sparsity structure depends on the dimensions alone) and compile the
  input generator and the repeated operation ahead of time from `jax.ShapeDtypeStruct`s.
  Nothing of the instance is generated here. The daemon (benchmark/server.py) does this in
  the untimed prepare_instance.sh, as the other JAX entries do.
- `run_instance(...)` is the measured part: generate the inputs on the device, then perform
  the operation `repetition` times, and write the verdict.

Without a daemon, run_instance.sh calls this file directly
(`python cora_comp.py <params-json> <result-file>`), and compilation happens lazily inside the
measurement. See https://github.com/CORA-COMP/benchmarks for the operations.

Batched benchmarks (`batch_size` present) apply the operation to the whole batch in one
`jax.vmap`ped call; unbatched ones call the library directly. How a batched CPU instance uses the
cores is configured in benchmark/config.env (see there).
"""
import json
import sys
import time

import jax
import jax.numpy as jnp

import csets
from csets import Interval, Zonotope

FINISHED, UNSUPPORTED, ERROR = "finished", "unsupported", "error"
SEED = 0  # every instance starts from the same key, so a warm daemon behaves like a fresh process


def write_result(path: str, verdict: str, **extra: float | int) -> None:
    columns = ["result", *extra]
    values = [verdict, *(f"{v:.6f}" if isinstance(v, float) else str(v) for v in extra.values())]
    with open(path, "w") as f:
        f.write(",".join(columns) + "\n" + ",".join(values) + "\n")


def unit_direction(key, dim: int):
    d = jax.random.normal(key, (dim,))
    return d / jnp.linalg.norm(d)


def per_set(params: dict):
    """
    The instance as seen by one set: `make(key)` its inputs, `shared(key)` inputs common to the
    whole batch (matMul's matrix), and `apply(key, inputs, shared)` one repetition on it.
    """
    operation = params["operation"]
    dim = params["dim"]
    points = params.get("points")

    def random_set(key):
        if params["set"] == "interval":
            return Interval.random(key, dim=dim)
        return Zonotope.random(key, dim=dim, nr_generators=params["generators"])

    def two(key, first, second):
        k1, k2 = jax.random.split(key)
        return first(k1), second(k2)

    def nothing(_):
        return ()

    if operation in ("startup", "generateRandom"):
        return nothing, nothing, lambda key, inputs, shared: random_set(key)
    if operation == "randPoint":
        return random_set, nothing, lambda key, s, shared: s.sample(key, points)
    if operation == "supportFunc":
        return (lambda key: two(key, random_set, lambda k: unit_direction(k, dim)), nothing,
                lambda key, inputs, shared: inputs[0].support(inputs[1]))
    if operation == "matMul":
        # one matrix for the whole batch, per the catalog
        return random_set, lambda key: jax.random.normal(key, (dim, dim)), lambda key, s, matrix: matrix @ s
    if operation == "minkSum":
        return (lambda key: two(key, random_set, random_set), nothing,
                lambda key, inputs, shared: inputs[0].minkowski_sum(inputs[1]))
    if operation == "contains":
        # The containment check depends on the set's type and shape only, so it is set up from an
        # example of them, not from the instance's sets.
        solver = random_set(jax.random.PRNGKey(SEED)).make_contains(jnp.zeros(dim))

        def make(key):
            k1, k2 = jax.random.split(key)
            s = random_set(k1)
            return s, s.sample(k2, points)
        # vmap over the points: they share the zonotope's factorisation, as in one call
        return make, nothing, lambda key, inputs, shared: jax.vmap(lambda p: inputs[0].contains(p, solver))(inputs[1])
    raise ValueError(f"Unknown operation '{operation}'")


def programs(params: dict, set_program=None, stream: int = 0):
    """
    Return `(generate, operation)` for the instance, both still to be jitted: `generate()` makes
    the inputs, and `operation(i, inputs)` is repetition `i`, which folds `i` into its keys so
    random operations draw fresh numbers each time. A batched instance is the per-set program
    under `jax.vmap`.

    Args:
        params: The instance; for one shard of a split batch, with that shard's `batch_size`.
        set_program: `per_set(params)`, when already built (the shards of a batch share it).
        stream: Folded into every key, so that each shard of a split batch draws its own sets.
    """
    make, shared, apply = set_program or per_set(params)
    batch = params.get("batch_size")  # absent on the unbatched benchmarks
    set_key, shared_key, rep_key = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(SEED), stream), 3)

    if batch is None:
        def generate():
            return make(set_key), shared(shared_key)

        def operation(i, inputs):
            return apply(jax.random.fold_in(rep_key, i), inputs[0], inputs[1])
        return generate, operation

    def generate():
        return jax.random.split(rep_key, batch), jax.vmap(make)(jax.random.split(set_key, batch)), shared(shared_key)

    def operation(i, inputs):
        keys, sets, common = inputs
        return jax.vmap(lambda k, s: apply(jax.random.fold_in(k, i), s, common))(keys, sets)
    return generate, operation


MAX_SHARDS = 20  # most shards measured: each costs a compilation in prepare and a dispatch per repetition


def shard_counts(params: dict) -> list[int]:
    """
    The numbers of shards worth measuring for an instance. Only a batched CPU instance on a CPU split
    into devices (see benchmark/config.env) has more than one: besides 1, the most that divide the
    batch (at most `MAX_SHARDS`) and one between, near their square root, since the split's gain and
    its overhead both grow with the number of shards.
    """
    batch = params.get("batch_size")
    if params["device"] != "cpu" or batch is None:
        return [1]
    limit = min(len(jax.devices("cpu")), batch, MAX_SHARDS)
    divisors = [k for k in range(1, limit + 1) if batch % k == 0]
    between = max(k for k in divisors if k * k <= divisors[-1])
    return sorted({1, between, divisors[-1]})


def configure(params: dict):
    """Return the JAX device for the instance (None if there is none) and switch off gradients."""
    try:
        device = jax.devices("cuda" if params["device"] == "gpu" else "cpu")[0]
    except RuntimeError:
        return None
    # Moreau's layers follow JAX's default device, which Instance sets to this one while building.
    csets.config.enable_grad = False
    return device


class Instance:
    """
    One catalog instance: shape-only setup in the constructor and `compile`, the measured part in `run`.

    A batched CPU instance may be split into shards, one per XLA CPU device (see benchmark/config.env):
    each device gets its slice of the batch and its own compiled copy of the ordinary batched
    program, and every repetition is dispatched to all of them, which run in parallel. Splitting
    by hand rather than by `shard_map` keeps Moreau's host callback (the containment fallback)
    out of a sharded program, where it crashes XLA.

    Splitting costs a fixed overhead per shard and repetition, and only pays off for operations that
    do enough work per repetition, so `compile` measures a few splits and keeps the fastest; see there.
    Without `compile`, the instance runs unsplit.
    """

    def __init__(self, params: dict):
        self.params = params
        self.unsupported = None
        if params["set"] not in ("zonotope", "interval"):
            self.unsupported = f"csets has no {params['set']} representation"
            return
        self.device = configure(params)
        if self.device is None:
            self.unsupported = f"no {params['device']} device available to JAX"
            return
        with jax.default_device(self.device):
            # Moreau's JAX bindings find their solver through a weak registry, and a compiled
            # program holds only the solver's id; this reference is what keeps the solver (the
            # containment check's LP fallback) alive.
            self._set_program = per_set(params)
        self.choice = None  # how compile() chose the number of shards, for the log
        self.shards, self.devices, self.generate, self.operation = self._programs(1)

    def _programs(self, shards: int) -> tuple[int, list, list, list]:
        """The instance's programs split into this many shards, jitted but not yet compiled."""
        devices = jax.devices("cpu")[:shards] if shards > 1 else [self.device]
        shard_params = dict(self.params, batch_size=self.params["batch_size"] // shards) if shards > 1 else self.params
        generates, operations = [], []
        for k, device in enumerate(devices):
            with jax.default_device(device):
                generate, operation = programs(shard_params, self._set_program, stream=k)
            generates.append(jax.jit(generate, device=device) if shards > 1 else jax.jit(generate))
            operations.append(jax.jit(operation, device=device) if shards > 1 else jax.jit(operation))
        return shards, devices, generates, operations

    def compile(self) -> None:
        """
        Compile the programs from shapes alone. A batched CPU instance that could be split is also
        compiled split into each of `shard_counts`, and each is timed on generated inputs, which are
        discarded. A split is only tried if a repetition takes at least a millisecond unsplit (below
        that, a split's overhead dominates and the difference is within the measurement's noise), and
        only kept if it is at least 10% faster than unsplit; the fastest such split wins. This is
        untimed setup, like the compilation itself: it chooses how to run the instance, from the
        instance's shape and the machine, and measures nothing of the run.
        """
        if self.unsupported:
            return
        chosen = (self.shards, self.devices, self.generate, self.operation)  # unsplit, from __init__
        self._compile_programs(chosen)
        counts = shard_counts(self.params)
        if len(counts) > 1:
            unsplit = fastest = self._seconds_per_repetition(chosen)
            measured = [f"1 shard(s): {1e3 * unsplit:.3f} ms"]
            if unsplit >= 1e-3:
                for shards in counts[1:]:
                    candidate = self._programs(shards)
                    self._compile_programs(candidate)
                    seconds = self._seconds_per_repetition(candidate)
                    measured.append(f"{shards} shard(s): {1e3 * seconds:.3f} ms")
                    if seconds < min(fastest, 0.9 * unsplit):
                        chosen, fastest = candidate, seconds
            self.choice = ", ".join(measured)
        self.shards, self.devices, self.generate, self.operation = chosen

    def _compile_programs(self, programs: tuple[int, list, list, list]) -> None:
        """Compile one split's programs from shapes alone, in place."""
        _, devices, generates, operations = programs
        for k, device in enumerate(devices):
            with jax.default_device(device):
                generates[k] = generates[k].lower().compile()
                operations[k] = operations[k].lower(0, generates[k].out_info).compile()

    def _seconds_per_repetition(self, programs: tuple[int, list, list, list]) -> float:
        """
        Time the compiled programs on inputs generated for the purpose: one repetition to warm up,
        then repetitions dispatched back to back, as `run` does, until they take 0.1 s (at most
        100); if one repetition already takes over a second, that one is the estimate.
        """
        _, _, generates, operations = programs
        with jax.default_device(self.device):
            inputs = [generate() for generate in generates]
            start = time.perf_counter()
            jax.block_until_ready([operation(0, x) for operation, x in zip(operations, inputs)])
            first = time.perf_counter() - start
            if first > 1:
                return first
            repetitions = max(3, min(100, int(0.1 / max(first, 1e-6))))
            start = time.perf_counter()
            for i in range(1, repetitions + 1):
                outputs = [operation(i, x) for operation, x in zip(operations, inputs)]
            jax.block_until_ready(outputs)
            return (time.perf_counter() - start) / repetitions

    def run(self):
        """
        Generate the inputs, then repeat the operation; return (time_generate, time_operation, outputs),
        the last repetition's output of each shard. They stay where they were computed: gathering them
        would add a copy of the whole result to the measurement.
        """
        with jax.default_device(self.device):
            t0 = time.perf_counter()
            inputs = [generate() for generate in self.generate]
            jax.block_until_ready(inputs)
            t1 = time.perf_counter()
            for i in range(self.params["repetition"]):
                # dispatched asynchronously: the shards' devices work on them at the same time
                outputs = [operation(i, x) for operation, x in zip(self.operation, inputs)]  # overwritten, like `Z2 = M * Z` in CORA
            jax.block_until_ready(outputs)  # like wait(gpuDevice): calls return before the work is done
            t2 = time.perf_counter()
        return t1 - t0, t2 - t1, outputs


def run_instance(instance: Instance, result_file: str) -> str:
    """The measured part: run the instance and write its verdict, which is also returned."""
    params = instance.params
    if instance.unsupported:
        print(f"{instance.unsupported}; reporting unsupported.")
        write_result(result_file, UNSUPPORTED)
        return UNSUPPORTED

    time_generate, time_operation, outputs = instance.run()

    verdict = FINISHED
    if params["operation"] == "contains":
        # Every point was drawn from its set, so every answer must be true.
        n_true, n = sum(int(answers.sum()) for answers in outputs), sum(answers.size for answers in outputs)
        print(f"contains: {n_true}/{n} true")
        if n_true != n:
            verdict = ERROR

    write_result(result_file, verdict, time_generate=time_generate, time_operation=time_operation,
                 shards=instance.shards)
    print(f"{verdict}: generate {time_generate:.3f}s, operation {time_operation:.3f}s ({params['repetition']} reps)")
    return verdict


def main() -> int:
    run_instance(Instance(json.loads(sys.argv[1])), sys.argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
