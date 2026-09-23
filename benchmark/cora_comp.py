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
from csets import Zonotope

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
    dim, generators = params["dim"], params["generators"]
    points = params.get("points")

    def zonotope(key):
        return Zonotope.random(key, dim=dim, nr_generators=generators)

    def two(key, first, second):
        k1, k2 = jax.random.split(key)
        return first(k1), second(k2)

    def nothing(_):
        return ()

    if operation in ("startup", "generateRandom"):
        return nothing, nothing, lambda key, inputs, shared: zonotope(key)
    if operation == "randPoint":
        return zonotope, nothing, lambda key, z, shared: z.sample(key, points)
    if operation == "supportFunc":
        return (lambda key: two(key, zonotope, lambda k: unit_direction(k, dim)), nothing,
                lambda key, inputs, shared: inputs[0].support(inputs[1]))
    if operation == "matMul":
        # one matrix for the whole batch, per the catalog
        return zonotope, lambda key: jax.random.normal(key, (dim, dim)), lambda key, z, matrix: matrix @ z
    if operation == "minkSum":
        return (lambda key: two(key, zonotope, zonotope), nothing,
                lambda key, inputs, shared: inputs[0].minkowski_sum(inputs[1]))
    if operation == "contains":
        # The containment check depends on the shapes only, so it is set up from zeros of the
        # instance's shape, not from the instance's sets. It takes all points of a set at once.
        example = Zonotope(centre=jnp.zeros(dim), generator=jnp.zeros((dim, generators)))
        solver = example.make_contains(jnp.zeros((points, dim)))

        def make(key):
            k1, k2 = jax.random.split(key)
            z = zonotope(k1)
            return z, z.sample(k2, points)
        return make, nothing, lambda key, inputs, shared: inputs[0].contains(inputs[1], solver)
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


def shards_for(params: dict) -> int:
    """
    How many CPU devices a batched CPU instance is split across: the most that divide the batch.
    There is more than one only when benchmark/config.env splits the CPU into devices.
    """
    batch = params.get("batch_size")
    if params["device"] != "cpu" or batch is None:
        return 1
    devices = len(jax.devices("cpu"))
    return max(k for k in range(1, min(devices, batch) + 1) if batch % k == 0)


def configure(params: dict):
    """Return the JAX device for the instance (None if there is none) and pin csets to it."""
    try:
        device = jax.devices("cuda" if params["device"] == "gpu" else "cpu")[0]
    except RuntimeError:
        return None
    # Pin the Moreau solver to the instance's device too; by default csets picks by problem size.
    csets.config.device = "cuda" if params["device"] == "gpu" else "cpu"
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
    """

    def __init__(self, params: dict):
        self.params = params
        self.unsupported = None
        if params["set"] != "zonotope":
            self.unsupported = f"csets has no {params['set']} representation"
            return
        self.device = configure(params)
        if self.device is None:
            self.unsupported = f"no {params['device']} device available to JAX"
            return
        self.shards = shards_for(params)
        self.devices = jax.devices("cpu")[:self.shards] if self.shards > 1 else [self.device]
        shard_params = dict(params, batch_size=params["batch_size"] // self.shards) if self.shards > 1 else params
        with jax.default_device(self.device):
            set_program = per_set(params)
        # Moreau's JAX bindings find their solver through a weak registry, and a compiled program
        # holds only the solver's id; this reference is what keeps the solver (the containment
        # check's LP fallback) alive.
        self._set_program = set_program
        self.generate, self.operation = [], []
        for k, device in enumerate(self.devices):
            with jax.default_device(device):
                generate, operation = programs(shard_params, set_program, stream=k)
            self.generate.append(jax.jit(generate, device=device) if self.shards > 1 else jax.jit(generate))
            self.operation.append(jax.jit(operation, device=device) if self.shards > 1 else jax.jit(operation))

    def compile(self) -> None:
        """Compile every shard's programs from shapes alone; no input is generated."""
        if self.unsupported:
            return
        for k, device in enumerate(self.devices):
            with jax.default_device(device):
                self.generate[k] = self.generate[k].lower().compile()
                self.operation[k] = self.operation[k].lower(0, self.generate[k].out_info).compile()

    def run(self):
        """Generate the inputs, then repeat the operation; return (time_generate, time_operation, output)."""
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
        output = outputs[0] if len(outputs) == 1 else jax.tree.map(lambda *xs: jnp.concatenate([jax.device_get(x) for x in xs]), *outputs)
        return t1 - t0, t2 - t1, output


def run_instance(instance: Instance, result_file: str) -> str:
    """The measured part: run the instance and write its verdict, which is also returned."""
    params = instance.params
    if instance.unsupported:
        print(f"{instance.unsupported}; reporting unsupported.")
        write_result(result_file, UNSUPPORTED)
        return UNSUPPORTED

    time_generate, time_operation, output = instance.run()

    verdict = FINISHED
    if params["operation"] == "contains":
        # Every point was drawn from its set, so every answer must be true.
        n_true = int(output.sum())
        print(f"contains: {n_true}/{output.size} true")
        if n_true != output.size:
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
