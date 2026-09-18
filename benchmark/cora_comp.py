"""
CORA-COMP driver: runs one benchmark instance with csets and writes the verdict.

Called by run_instance.sh as `python cora_comp.py <params-json> <result-file>`. Everything
here is timed by the harness, so the script does exactly what the catalog defines: generate
the inputs, move them to the device, then perform the operation `repetition` times.
See https://github.com/CORA-COMP/benchmarks for the operations.

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


def build(params: dict, key):
    """
    Return (inputs, op) for the instance: `inputs` is generated before the loop and `op(key, *inputs)`
    is the repeated call; `key` differs per repetition so random operations draw fresh samples.
    """
    operation = params["operation"]
    dim, generators = params["dim"], params["generators"]
    batch = params.get("batch_size")  # absent on the unbatched benchmarks
    k1, k2, k3 = jax.random.split(key, 3)

    if operation == "startup":
        return (), lambda k: Zonotope.random(k, dim=dim, nr_generators=generators)

    if operation == "generateRandom":
        return (), lambda k: random_zonotopes(k, batch, dim, generators)

    if operation == "randPoint":
        points = params["points"]
        z = random_zonotopes(k1, batch, dim, generators)
        if batch is None:
            return (z,), lambda k, z: z.sample(k, points)
        return (z,), lambda k, z: jax.vmap(lambda zi, ki: zi.sample(ki, points))(z, jax.random.split(k, batch))

    if operation == "supportFunc":
        z = random_zonotopes(k1, batch, dim, generators)
        d = unit_directions(k2, batch, dim)
        if batch is None:
            return (z, d), lambda k, z, d: z.support(d)
        return (z, d), lambda k, z, d: jax.vmap(lambda zi, di: zi.support(di))(z, d)

    if operation == "matMul":
        z = random_zonotopes(k1, batch, dim, generators)
        m = jax.random.normal(k2, (dim, dim))  # one matrix for the whole batch, per the catalog
        if batch is None:
            return (m, z), lambda k, m, z: m @ z
        return (m, z), lambda k, m, z: jax.vmap(lambda zi: m @ zi)(z)

    if operation == "minkSum":
        z1 = random_zonotopes(k1, batch, dim, generators)
        z2 = random_zonotopes(k2, batch, dim, generators)
        if batch is None:
            return (z1, z2), lambda k, z1, z2: z1.minkowski_sum(z2)
        return (z1, z2), lambda k, z1, z2: jax.vmap(lambda a, b: a.minkowski_sum(b))(z1, z2)

    if operation == "contains":
        points = params["points"]
        z = random_zonotopes(k1, batch, dim, generators)
        if batch is None:
            p = z.sample(k2, points)
            solver = z.make_contains(p[0])
            return (z, p), lambda k, z, p: jax.vmap(lambda pi: z.contains(pi, solver))(p)
        # `points` points per set, drawn from that set. Moreau batches over one leading axis,
        # so the (batch, points) pairs are flattened into one batch of batch*points solves.
        p = jax.vmap(lambda zi, ki: zi.sample(ki, points))(z, jax.random.split(k2, batch))
        solver = jax.tree.map(lambda x: x[0], z).make_contains(p[0, 0])
        z_flat = jax.tree.map(lambda x: jnp.repeat(x, points, axis=0), z)
        p_flat = p.reshape(batch * points, dim)
        return (z_flat, p_flat), lambda k, z, p: jax.vmap(lambda zi, pi: zi.contains(pi, solver))(z, p)

    raise ValueError(f"Unknown operation '{operation}'")


def main() -> int:
    params = json.loads(sys.argv[1])
    result_file = sys.argv[2]

    if params["set"] != "zonotope":
        print(f"csets has no {params['set']} representation; reporting unsupported.")
        write_result(result_file, UNSUPPORTED)
        return 0

    device_kind = params["device"]
    try:
        device = jax.devices("cuda" if device_kind == "gpu" else "cpu")[0]
    except RuntimeError:
        print(f"No {device_kind} device available to JAX; reporting unsupported.")
        write_result(result_file, UNSUPPORTED)
        return 0

    # Pin the Moreau solver to the instance's device too; by default csets picks by problem size.
    csets.config.device = "cuda" if device_kind == "gpu" else "cpu"
    csets.config.enable_grad = False

    repetition = params["repetition"]
    key = jax.random.PRNGKey(0)
    setup_key, *rep_keys = jax.random.split(key, repetition + 1)

    with jax.default_device(device):
        t0 = time.perf_counter()
        inputs, op = build(params, setup_key)
        inputs = jax.device_put(inputs, device)
        jax.block_until_ready(inputs)
        t1 = time.perf_counter()

        op = jax.jit(op)
        for i in range(repetition):
            output = op(rep_keys[i], *inputs)  # overwritten each time, like `Z2 = M * Z` in CORA
        jax.block_until_ready(output)  # like wait(gpuDevice): calls return before the work is done
        t2 = time.perf_counter()

    if params["operation"] == "contains":
        # Every point was drawn from the set, so every answer must be true.
        print(f"contains: {int(output.sum())}/{output.size} true")

    write_result(result_file, FINISHED, time_generate=t1 - t0, time_operation=t2 - t1)
    print(f"finished: generate {t1 - t0:.3f}s, operation {t2 - t1:.3f}s ({repetition} reps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
