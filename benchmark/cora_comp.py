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
`jax.vmap`ped call; unbatched ones call the library directly.
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


def write_result(path: str, verdict: str, **extra: float) -> None:
    columns = ["result", *extra]
    values = [verdict, *(f"{v:.6f}" for v in extra.values())]
    with open(path, "w") as f:
        f.write(",".join(columns) + "\n" + ",".join(values) + "\n")


def random_zonotopes(key, batch: int | None, dim: int, generators: int):
    """A random zonotope, or a batch of them (leading axis `batch`) when batched."""
    if batch is None:
        return Zonotope.random(key, dim=dim, nr_generators=generators)
    return jax.vmap(lambda k: Zonotope.random(k, dim=dim, nr_generators=generators))(jax.random.split(key, batch))


def unit_directions(key, batch: int | None, dim: int):
    shape = (dim,) if batch is None else (batch, dim)
    d = jax.random.normal(key, shape)
    return d / jnp.linalg.norm(d, axis=-1, keepdims=True)


def programs(params: dict):
    """
    Return `(generate, operation)` for the instance, both still to be jitted:
    `generate()` makes the inputs, and `operation(i, inputs)` is repetition `i`, which folds `i`
    into its key so random operations draw fresh numbers each time.
    """
    operation = params["operation"]
    dim, generators = params["dim"], params["generators"]
    batch = params.get("batch_size")  # absent on the unbatched benchmarks

    def input_keys():
        return jax.random.split(jax.random.PRNGKey(SEED), 2)

    def rep_key(i):
        return jax.random.fold_in(jax.random.PRNGKey(SEED + 1), i)

    if operation == "startup":
        return (lambda: ()), lambda i, _: Zonotope.random(rep_key(i), dim=dim, nr_generators=generators)

    if operation == "generateRandom":
        return (lambda: ()), lambda i, _: random_zonotopes(rep_key(i), batch, dim, generators)

    if operation == "randPoint":
        points = params["points"]

        def generate():
            return (random_zonotopes(input_keys()[0], batch, dim, generators),)

        if batch is None:
            return generate, lambda i, inputs: inputs[0].sample(rep_key(i), points)
        return generate, lambda i, inputs: jax.vmap(lambda zi, ki: zi.sample(ki, points))(
            inputs[0], jax.random.split(rep_key(i), batch))

    if operation == "supportFunc":
        def generate():
            k1, k2 = input_keys()
            return random_zonotopes(k1, batch, dim, generators), unit_directions(k2, batch, dim)

        if batch is None:
            return generate, lambda i, inputs: inputs[0].support(inputs[1])
        return generate, lambda i, inputs: jax.vmap(lambda zi, di: zi.support(di))(*inputs)

    if operation == "matMul":
        def generate():
            k1, k2 = input_keys()
            # one matrix for the whole batch, per the catalog
            return jax.random.normal(k2, (dim, dim)), random_zonotopes(k1, batch, dim, generators)

        if batch is None:
            return generate, lambda i, inputs: inputs[0] @ inputs[1]
        return generate, lambda i, inputs: jax.vmap(lambda zi: inputs[0] @ zi)(inputs[1])

    if operation == "minkSum":
        def generate():
            k1, k2 = input_keys()
            return random_zonotopes(k1, batch, dim, generators), random_zonotopes(k2, batch, dim, generators)

        if batch is None:
            return generate, lambda i, inputs: inputs[0].minkowski_sum(inputs[1])
        return generate, lambda i, inputs: jax.vmap(lambda a, b: a.minkowski_sum(b))(*inputs)

    if operation == "contains":
        points = params["points"]
        # The solver's structure depends on the shapes only, so it is built from zeros of the
        # instance's shape, not from the instance's sets.
        example = Zonotope(centre=jnp.zeros(dim), generator=jnp.zeros((dim, generators)))
        solver = example.make_contains(jnp.zeros(dim))

        if batch is None:
            def generate():
                k1, k2 = input_keys()
                z = random_zonotopes(k1, batch, dim, generators)
                return z, z.sample(k2, points)

            return generate, lambda i, inputs: jax.vmap(lambda pi: inputs[0].contains(pi, solver))(inputs[1])

        # `points` points per set, drawn from that set. Moreau batches over one leading axis,
        # so the (batch, points) pairs are flattened into one batch of batch*points solves.
        def generate():
            k1, k2 = input_keys()
            z = random_zonotopes(k1, batch, dim, generators)
            p = jax.vmap(lambda zi, ki: zi.sample(ki, points))(z, jax.random.split(k2, batch))
            z_flat = jax.tree.map(lambda x: jnp.repeat(x, points, axis=0), z)
            return z_flat, p.reshape(batch * points, dim)

        return generate, lambda i, inputs: jax.vmap(lambda zi, pi: zi.contains(pi, solver))(*inputs)

    raise ValueError(f"Unknown operation '{operation}'")


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
    """One catalog instance: shape-only setup in the constructor and `compile`, the measured part in `run`."""

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
        with jax.default_device(self.device):
            generate, operation = programs(params)
        # Moreau's JAX bindings find their solver through a weak registry, and a compiled program
        # holds only the solver's id; these closures are what keeps the solver alive.
        self._programs = generate, operation
        self.generate, self.operation = jax.jit(generate), jax.jit(operation)

    def compile(self) -> None:
        """Compile both programs from shapes alone; no input is generated."""
        if self.unsupported:
            return
        with jax.default_device(self.device):
            lowered = self.generate.lower()
            shapes = lowered.out_info
            self.generate = lowered.compile()
            self.operation = self.operation.lower(0, shapes).compile()

    def run(self):
        """Generate the inputs, then repeat the operation; return (time_generate, time_operation, output)."""
        with jax.default_device(self.device):
            t0 = time.perf_counter()
            inputs = self.generate()
            jax.block_until_ready(inputs)
            t1 = time.perf_counter()
            for i in range(self.params["repetition"]):
                output = self.operation(i, inputs)  # overwritten each time, like `Z2 = M * Z` in CORA
            jax.block_until_ready(output)  # like wait(gpuDevice): calls return before the work is done
            t2 = time.perf_counter()
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

    write_result(result_file, verdict, time_generate=time_generate, time_operation=time_operation)
    print(f"{verdict}: generate {time_generate:.3f}s, operation {time_operation:.3f}s ({params['repetition']} reps)")
    return verdict


def main() -> int:
    run_instance(Instance(json.loads(sys.argv[1])), sys.argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
