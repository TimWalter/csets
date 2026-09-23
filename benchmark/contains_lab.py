"""
Lab for zonotope point containment on the CORA-COMP `contains` instances: which method, and
which solver settings, answer fastest, and do they still answer correctly.

    python benchmark/contains_lab.py --device cpu                      # speed sweep, all methods
    python benchmark/contains_lab.py --device gpu --methods moreau-ipm-cudss cert
    python benchmark/contains_lab.py --check                            # correctness vs HiGHS
    python benchmark/contains_lab.py --list

Speed: for every method, dimension and batch size it builds the instance exactly like the
benchmark (random zonotopes, `points` points drawn from each), compiles, and reports the
steady-state time of one repetition, which is what the harness multiplies by 100. A method
stops growing in dimension once a repetition exceeds `--budget` seconds (100 of those would
not fit the 60 s timeout anyway).

Correctness (`--check`): points inside, on the way out and outside (the benchmark only ever
asks about points inside, so a solver that says "true" to everything would pass it), against
SciPy's HiGHS. Points whose LP value lies within 1e-6 of the boundary are not counted.

Methods:
- `moreau-*`: the LP of csets' `_contains_point` (min ||beta||_inf s.t. G beta = p - c) with
  different Moreau settings; `moreau-box-*` asks for feasibility in the box instead.
- `linrax`: the same LP, split into non-negative variables, on linrax's simplex.
- `cert`: no LP for most points. Facets while C(m, n-1) is small, otherwise alternating
  projections between {beta | G beta = p - c} and a shrunk box, which certify "inside"
  (a beta with ||beta||_inf <= 1) or "outside" (a separating direction); only points neither
  settles go to the LP. This is what CORA.cpp / CORA.py / CORA.jax do.
"""
import argparse
import itertools
import math
import os
import sys
import time
from pathlib import Path

DIMS = (1, 2, 5, 10, 50, 100, 500, 1000)
BATCHES = (None, 10, 100)  # None: the unbatched benchmark
POINTS = 10
TOL = 1e-6


# --- methods ------------------------------------------------------------------------------------
# Every builder takes (d, m, device) and returns (fn, keepalive); fn maps centres (B, d),
# generators (B, d, m) and points (B, N, d) to a (B, N) mask and is jitted by the caller.

def moreau_settings(solver="ipm", method=None, tol=None, equil=True, yolo=None, max_iter=None):
    def make(device):
        from moreau.jax import Settings
        from moreau._types import IPMSettings
        ipm = IPMSettings()
        if method:
            ipm.direct_solve_method = method
        if tol:
            ipm.tol_gap_abs = ipm.tol_gap_rel = ipm.tol_feas = tol
        ipm.equilibrate_enable = equil
        s = Settings(solver=solver, device="cuda" if device == "gpu" else "cpu", ipm_settings=ipm)
        if yolo:
            s.yolo, s.yolo_num_iters = True, yolo
        if max_iter:
            s.max_iter = max_iter
        return s
    return make


def lp_structure(d, m, box):
    """CSR structure of csets' containment LP in Moreau's form A z + s = b, s in K."""
    import jax.numpy as jnp
    if box:  # z = beta; G beta = r; beta <= 1; -beta <= 1
        n_var = m
        row_offsets = jnp.concat([jnp.arange(0, d * m + 1, m), d * m + jnp.arange(1, 2 * m + 1)])
        cols = jnp.concat([jnp.tile(jnp.arange(m), d), jnp.arange(m), jnp.arange(m)])
    else:  # z = [beta, t]; G beta = r; beta - t <= 0; -beta - t <= 0   (csets' _make_contains_point)
        n_var = m + 1
        row_offsets = jnp.concat([jnp.arange(0, d * m + 1, m), d * m + jnp.arange(2, 4 * m + 1, 2)])
        bound = jnp.stack([jnp.arange(m), jnp.full(m, m)], axis=1).flatten()
        cols = jnp.concat([jnp.tile(jnp.arange(m), d), bound, bound])
    return n_var, d + 2 * m, row_offsets.astype(jnp.int32), cols.astype(jnp.int32)


def lp_values(G, r, box):
    import jax.numpy as jnp
    m = G.shape[1]
    if box:
        a = jnp.concat([G.flatten(), jnp.ones(m), -jnp.ones(m)])
        return a, jnp.zeros(m), jnp.concat([r, jnp.ones(2 * m)])
    a = jnp.concat([G.flatten(), jnp.tile(jnp.array([1.0, -1.0]), m), jnp.full(2 * m, -1.0)])
    return a, jnp.zeros(m + 1).at[-1].set(1.0), jnp.concat([r, jnp.zeros(2 * m)])


def lp_decide(x, G, r, box):
    """Inside iff the solver's point is a certificate: G beta = r and ||beta||_inf <= 1, up to TOL.
    Checking the returned point instead of trusting t also turns a failed solve into "outside"
    rather than a wrong "inside"."""
    import jax.numpy as jnp
    beta = x if box else x[:-1]
    residual = jnp.abs(G @ beta - r).max() <= TOL * (1 + jnp.abs(r).max())
    return residual & (jnp.abs(beta).max() <= 1 + TOL)


def moreau_method(settings, box=False, verify=True):
    def build(d, m, device):
        import jax
        import jax.numpy as jnp
        from moreau.jax import Cones, Solver
        n_var, n_con, row_offsets, cols = lp_structure(d, m, box)
        solver = Solver(n=n_var, m=n_con,
                        P_row_offsets=jnp.zeros(n_var + 1, dtype=jnp.int32), P_col_indices=jnp.array([], dtype=jnp.int32),
                        A_row_offsets=row_offsets, A_col_indices=cols,
                        cones=Cones(num_zero_cones=d, num_nonneg_cones=2 * m), settings=settings(device))

        def one(c, G, p):
            r = p - c
            x = solver.solve(jnp.array([]), *lp_values(G, r, box)).x
            return lp_decide(x, G, r, box) if verify else x[-1] <= 1 + TOL

        def fn(C, G, P):
            B, N = P.shape[:2]
            flat = jax.vmap(one)(jnp.repeat(C, N, 0), jnp.repeat(G, N, 0), P.reshape(B * N, -1))
            return flat.reshape(B, N)
        return fn, solver
    return build


def linrax_method(d, m, device):
    import jax
    import jax.numpy as jnp
    from linrax import linprog
    # z = [beta+, beta-, t] >= 0: G beta+ - G beta- = r, beta+ + beta- - t <= 0.
    cost = jnp.zeros(2 * m + 1).at[-1].set(1.0)
    A_ub = jnp.concat([jnp.eye(m), jnp.eye(m), -jnp.ones((m, 1))], axis=1)

    def one(c, G, p):
        r = p - c
        sol, kind = linprog(cost, A_ub=A_ub, b_ub=jnp.zeros(m),
                            A_eq=jnp.concat([G, -G, jnp.zeros((d, 1))], axis=1), b_eq=r)
        beta = sol.x[:m] - sol.x[m:2 * m]
        return kind.success[0] & lp_decide(beta, G, r, box=True)

    def fn(C, G, P):
        return jax.vmap(lambda c, g, ps: jax.vmap(lambda p: one(c, g, p))(ps))(C, G, P)
    return fn, None


def cvxpylayers_method(solver, **solver_args):
    """The same LP as csets (min t s.t. G beta = r, -t <= beta <= t), written in CVXPY and solved
    through a cvxpylayers JAX layer. G and r are parameters, so CVXPY canonicalises once per
    shape; every call only maps the parameter values into the solver. Only the MOREAU path
    runs inside jax.jit; the others (DIFFCP: SCS/ECOS/Clarabel, MPAX) are called eagerly."""
    def build(d, m, device):
        import cvxpy as cp
        import jax
        import jax.numpy as jnp
        from cvxpylayers.jax import CvxpyLayer
        G_par, r_par = cp.Parameter((d, m)), cp.Parameter(d)
        beta, t = cp.Variable(m), cp.Variable()
        problem = cp.Problem(cp.Minimize(t), [G_par @ beta == r_par, beta <= t, -beta <= t])
        args = dict(solver_args)
        if solver == "MOREAU":
            args.setdefault("device", "cuda" if device == "gpu" else "cpu")
        if solver == "CUCLARABEL":
            # cvxpylayers 1.2.0's JAX layer calls jax_solve_only, which only the CuClarabel
            # interface lacks; its jax_solve does the same solve and also returns backward data.
            from cvxpylayers.interfaces.cuclarabel_if import CUCLARABEL_data
            if not hasattr(CUCLARABEL_data, "jax_solve_only"):
                CUCLARABEL_data.jax_solve_only = lambda self, solver_args=None: self.jax_solve(solver_args)[:2]
            # Clarabel.jl 0.11.0 declares PythonCall and CUDA as hard dependencies, so Julia never
            # loads its PythonExt (the cupy -> CuArray bridge cvxpylayers needs) and 0.11.1 drops
            # it. Load the extension's code into Clarabel itself, where cvxpylayers looks next.
            from juliacall import Main as jl
            jl.seval("using Clarabel, PythonCall, CUDA, CUDA.CUSPARSE")
            if not jl.seval("isdefined(Clarabel, :cupy_to_cuvector)"):
                jl.seval('Base.include(Clarabel, joinpath(dirname(dirname(pathof(Clarabel))), "ext", "py2jl.jl"))')
        layer = CvxpyLayer(problem, parameters=[G_par, r_par], variables=[beta, t], solver=solver, solver_args=args)
        decide = jax.jit(jax.vmap(lambda b, g, r: lp_decide(b, g, r, box=True)))

        def fn(C, G, P):
            B, N = P.shape[:2]
            G_flat = jnp.repeat(G, N, 0)
            r = (P - C[:, None, :]).reshape(B * N, d)
            beta_opt, _ = layer(G_flat, r)
            return decide(beta_opt, G_flat, r).reshape(B, N)
        fn.eager = solver != "MOREAU"
        return fn, layer
    return build


# --- certificates -------------------------------------------------------------------------------

MAX_FACETS = 300
SHRINK = 0.9
MAX_ITER = 60
CHECK_EVERY = 4


def facet_normals(G, subsets, rows, signs):
    """Generalised cross products of every (n-1) generators: (..., F, n)."""
    import jax.numpy as jnp
    GS = jnp.moveaxis(G[..., subsets], -3, -2)          # (..., F, n, n-1)
    return jnp.linalg.det(GS[..., rows, :]) * signs


def by_facets(C, G, P):
    """(inside (B, N), settled (B, N)): exact while the zonotope is full-dimensional."""
    import jax.numpy as jnp
    import numpy as np
    d, m = G.shape[-2:]
    r = P - C[:, None, :]
    if d == 1:
        reach = jnp.abs(G).sum((-2, -1))
        return jnp.abs(r[..., 0]) <= reach[:, None] * (1 + TOL), jnp.broadcast_to((reach > 0)[:, None], r.shape[:2])
    subsets = np.array(list(itertools.combinations(range(m), d - 1)))
    rows = np.array([[j for j in range(d) if j != i] for i in range(d)])
    signs = np.array([(-1.0) ** i for i in range(d)])
    H = facet_normals(G, subsets, rows, signs)          # (B, F, d)
    reach = jnp.abs(H @ G).sum(-1)                      # (B, F)
    scale = jnp.abs(H).max(-1)
    valid = scale > 1e-12 * scale.max(-1, keepdims=True)
    within = jnp.abs(jnp.einsum("bnd,bfd->bnf", r, H)) <= reach[:, None, :] * (1 + TOL)
    inside = (within | ~valid[:, None, :]).all(-1)
    full = valid.any(-1)
    return inside, jnp.broadcast_to(full[:, None], inside.shape)


def by_projection(C, G, P):
    """(inside, settled), both (B, N), from alternating projections with certificates.

    p is in the zonotope iff the affine set A = {beta | G beta = r}, r = p - c, meets [-1, 1]^m.
    Alternating projections between A and the shrunk box [-SHRINK, SHRINK]^m reach a beta in A
    with ||beta||_inf <= 1 within a few steps for points well inside; the gap u between the two
    sets gives y = (G G')^-1 G u, which proves the opposite if y'r > ||G'y||_1.

    Projection onto A is x - G'(G G')^-1 (G x - r), with a Cholesky factor of G G'. No
    orthonormal basis is formed: a QR of G' costs 2.1 s at 1000d on XLA:CPU, and a triangular
    solve against all of G (d x 2d right-hand sides) 5 ms already at 50d, while solves with the
    N points as right-hand sides take microseconds. Both certificates are checked against G.
    Every product is one matrix-matrix product per set over all its N points."""
    import jax
    import jax.numpy as jnp
    Gt = jnp.swapaxes(G, -1, -2)                        # (B, m, d)
    factor = (jnp.linalg.cholesky(G @ Gt), True)        # G G' = L L'

    def gram_solve(v):                                  # (G G')^-1 v for v (B, N, d)
        return jnp.swapaxes(jax.scipy.linalg.cho_solve(factor, jnp.swapaxes(v, -1, -2)), -1, -2)

    r = P - C[:, None, :]                               # (B, N, d)
    rscale = 1 + jnp.abs(r).max(-1)
    eps = 1000 * jnp.finfo(P.dtype).eps

    def to_affine(x):                                   # projection onto A
        return x - gram_solve(x @ Gt - r) @ G

    def certify(x):
        """x is in A up to rounding; a certificate each way, both checked against G."""
        residual = jnp.abs(x @ Gt - r).max(-1) <= 1e-9 * rscale   # stricter than the LP solvers (1e-8)
        inside = residual & (jnp.abs(x).max(-1) <= 1 + eps)
        y = gram_solve((x - jnp.clip(x, -SHRINK, SHRINK)) @ Gt)   # (B, N, d)
        reach = jnp.abs(y @ G).sum(-1)                  # ||G'y||_1
        gap = (y * r).sum(-1) - reach
        outside = gap > 1e3 * eps * (reach + jnp.abs(y * r).sum(-1))
        return inside, outside & ~inside

    def cond(state):
        _, inside, outside, it = state
        return (it < MAX_ITER) & ~(inside | outside).all()

    def body(state):
        x, inside, outside, it = state
        x = jax.lax.fori_loop(0, CHECK_EVERY, lambda _, v: to_affine(jnp.clip(v, -SHRINK, SHRINK)), x)
        now_in, now_out = certify(x)
        return x, inside | now_in, outside | now_out, it + CHECK_EVERY

    x0 = gram_solve(r) @ G                              # least-norm solution
    _, inside, outside, _ = jax.lax.while_loop(cond, body, (x0, *certify(x0), 0))
    return inside, inside | outside


def cert_method(d, m, device):
    """Certificates in one compiled program; the few unsettled points go to an LP afterwards,
    host-driven, so the LP never runs (not even masked) for points that are settled."""
    fast = by_facets if d == 1 or math.comb(m, d - 1) <= MAX_FACETS else by_projection
    lp, keep = moreau_method(moreau_settings("ipm"))(d, m, device)
    import jax
    lp = jax.jit(lp)

    def fn(C, G, P):
        return fast(C, G, P)

    def finish(result, C, G, P):
        import numpy as np
        inside, settled = (np.array(a) for a in result)  # copies: JAX's buffers are read-only
        todo = np.argwhere(~settled)
        for b, k in todo:
            inside[b, k] = bool(lp(C[b:b + 1], G[b:b + 1], P[b:b + 1, k:k + 1])[0, 0])
        cert_method.unsettled += len(todo)
        return inside
    fn.finish = finish
    return fn, (keep, lp)


cert_method.unsettled = 0

METHODS = {
    "moreau-active_set (current)": moreau_method(moreau_settings("active_set"), verify=False),
    "moreau-active_set": moreau_method(moreau_settings("active_set")),
    "moreau-ipm": moreau_method(moreau_settings("ipm")),
    "moreau-ipm-qdldl": moreau_method(moreau_settings("ipm", "qdldl")),
    "moreau-ipm-faer-1t": moreau_method(moreau_settings("ipm", "faer-1t")),
    "moreau-ipm-faer": moreau_method(moreau_settings("ipm", "faer")),
    "moreau-ipm-cudss": moreau_method(moreau_settings("ipm", "cudss")),
    "moreau-ipm-tol1e-7": moreau_method(moreau_settings("ipm", tol=1e-7)),
    "moreau-ipm-noequil": moreau_method(moreau_settings("ipm", equil=False)),
    "moreau-ipm-yolo15": moreau_method(moreau_settings("ipm", yolo=15)),
    "moreau-ipm-yolo25": moreau_method(moreau_settings("ipm", yolo=25)),
    "moreau-box-ipm": moreau_method(moreau_settings("ipm"), box=True),
    "linrax": linrax_method,
    "cvxpylayers-scs": cvxpylayers_method("DIFFCP", solve_method="SCS", eps_abs=1e-7, eps_rel=1e-7),
    "cvxpylayers-ecos": cvxpylayers_method("DIFFCP", solve_method="ECOS"),
    "cvxpylayers-clarabel": cvxpylayers_method("DIFFCP", solve_method="Clarabel"),
    "cvxpylayers-moreau": cvxpylayers_method("MOREAU"),
    "cvxpylayers-mpax": cvxpylayers_method("MPAX", eps_abs=1e-7, eps_rel=1e-7),
    "cvxpylayers-cuclarabel": cvxpylayers_method("CUCLARABEL"),
    "cvxpylayers-mpax-r2hpdhg": cvxpylayers_method("MPAX", algorithm="r2HPDHG", eps_abs=1e-7, eps_rel=1e-7),
    "cert": cert_method,
}
CPU_ONLY = {"moreau-ipm-qdldl", "moreau-ipm-faer-1t", "moreau-ipm-faer",
            "cvxpylayers-scs", "cvxpylayers-ecos", "cvxpylayers-clarabel"}  # diffcp's solvers run on the host
GPU_ONLY = {"moreau-ipm-cudss", "cvxpylayers-cuclarabel"}


# --- driver -------------------------------------------------------------------------------------

def make_instance(key, d, m, batch, points, scale=None):
    """Like the benchmark: random zonotopes and points drawn from each. With `scale`, the points
    are c + scale * (p - c) instead, for correctness checks across the boundary."""
    import jax
    from csets import Zonotope
    B = batch or 1
    k1, k2 = jax.random.split(key)
    Z = jax.vmap(lambda k: Zonotope.random(k, dim=d, nr_generators=m))(jax.random.split(k1, B))
    P = jax.vmap(lambda z, k: z.sample(k, points))(Z, jax.random.split(k2, B))
    if scale is not None:
        P = Z.centre[:, None, :] + scale[None, :, None] * (P - Z.centre[:, None, :])
    return Z.centre, Z.generator, P


def run_method(name, d, batch, device, reps, points=POINTS, data=None):
    import jax
    fn, keep = METHODS[name](d, 2 * d, device)
    C, G, P = data if data is not None else make_instance(jax.random.PRNGKey(0), d, 2 * d, batch, points)
    finish = getattr(fn, "finish", None)
    jfn = fn if getattr(fn, "eager", False) else jax.jit(fn)

    def call():
        out = jfn(C, G, P)
        return finish(out, C, G, P) if finish else jax.block_until_ready(out)

    t = time.perf_counter()
    out = call()
    compile_time = time.perf_counter() - t
    t = time.perf_counter()
    n = 0
    while n < reps or (time.perf_counter() - t < 0.5 and n < 50):
        out = call()
        n += 1
    return (time.perf_counter() - t) / n, compile_time, out, keep


def cell(args, device):
    """One (method, dim, batch) measurement, run in a child process by `speed`; prints one JSON line."""
    import json
    import jax
    import numpy as np
    name, d, batch = args.cell[0], int(args.cell[1]), int(args.cell[2]) or None
    with jax.default_device(jax.devices("cuda" if device == "gpu" else "cpu")[0]):
        per_rep, _, out, _ = run_method(name, d, batch, device, args.reps)
    print(json.dumps({"per_rep": per_rep, "ok": bool(np.asarray(out).all()), "lp": cert_method.unsettled}))


def run_guarded(cmd, mem_limit, time_limit):
    """Run a child, killing it above `mem_limit` bytes resident or `time_limit` seconds.
    Returns (stdout, None) or (None, "mem" | "slow" | "crash")."""
    import subprocess
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    t0 = time.monotonic()
    while proc.poll() is None:
        try:
            rss = int(Path(f"/proc/{proc.pid}/status").read_text().split("VmRSS:")[1].split()[0]) * 1024
        except (OSError, IndexError, ValueError):
            rss = 0
        if rss > mem_limit or time.monotonic() - t0 > time_limit:
            proc.kill()
            proc.wait()
            return None, "mem" if rss > mem_limit else "slow"
        time.sleep(0.1)
    out = proc.stdout.read().strip().splitlines()
    return (out[-1], None) if proc.returncode == 0 and out else (None, "crash")


def speed(args, device):
    """Every cell in its own process under a memory and time watchdog, so a method that explodes
    at some size costs one cell, not the machine (a WSL VM that runs out of memory goes down)."""
    import json
    mem_total = int(Path("/proc/meminfo").read_text().split("MemTotal:")[1].split()[0]) * 1024
    mem_limit = args.mem_fraction * mem_total
    out_path = Path(args.out) if args.out else Path(__file__).resolve().parent / "local_results" / f"contains_lab_{device}_{time.strftime('%Y%m%d-%H%M%S')}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = open(out_path, "w")

    def emit(line):
        print(line, flush=True)
        out.write(line + "\n")
        out.flush()

    emit(f"seconds per repetition (the harness runs 100), device={device}; memory cap {mem_limit / 2**30:.1f} GiB, "
         f"{args.cell_timeout:.0f} s per cell")
    emit(f"'-' over budget at a lower dim, '!' a point drawn from its set reported outside, '*' LP fallback ran, "
         f"'mem'/'slow' stopped by the watchdog\n")
    emit(f"{'method':28s} {'batch':>5s}" + "".join(f"{d:>9d}d" for d in args.dims))
    batches = [b for b in BATCHES if (b or 1) in args.batch]
    for name in args.methods:
        if (device == "gpu" and name in CPU_ONLY) or (device == "cpu" and name in GPU_ONLY):
            continue
        for batch in batches:
            row, dead = f"{name:28s} {str(batch or 1):>5s}", False
            for d in args.dims:
                if dead:
                    row += f"{'-':>10s}"
                    continue
                cmd = [sys.executable, __file__, "--device", device, "--reps", str(args.reps),
                       "--cell", name, str(d), str(batch or 0)]
                result, stopped = run_guarded(cmd, mem_limit, args.cell_timeout)
                if stopped:
                    text, dead = stopped, True
                else:
                    r = json.loads(result)
                    text = f"{r['per_rep']:.4f}" + ("" if r["ok"] else "!") + ("*" if r["lp"] else "")
                    dead = r["per_rep"] > args.budget
                row += f"{text:>10s}"
            emit(row)
    emit(f"\nwritten to {out_path}")


def check(args, device):
    """Mixed inside/outside points vs HiGHS."""
    import jax
    import numpy as np
    from scipy.optimize import linprog
    scales = np.array([0.5, 0.9, 0.99, 1.01, 1.1, 1.5, 2.0, 3.0])
    print(f"correctness vs HiGHS on points c + s (p - c), s in {list(scales)}, device={device}\n")
    for d in [x for x in args.dims if x <= args.check_max_dim]:
        m = 2 * d
        data = make_instance(jax.random.PRNGKey(1), d, m, 4, len(scales), jax.numpy.asarray(scales))
        C, G, P = (np.asarray(a) for a in data)
        truth, ambiguous = np.zeros(P.shape[:2], bool), np.zeros(P.shape[:2], bool)
        for b, k in itertools.product(range(P.shape[0]), range(P.shape[1])):
            res = linprog(np.r_[np.zeros(m), 1.0],
                          A_ub=np.block([[np.eye(m), -np.ones((m, 1))], [-np.eye(m), -np.ones((m, 1))]]), b_ub=np.zeros(2 * m),
                          A_eq=np.c_[G[b], np.zeros(d)], b_eq=P[b, k] - C[b], bounds=[(None, None)] * m + [(0, None)])
            truth[b, k] = res.status == 0 and res.fun <= 1
            ambiguous[b, k] = res.status == 0 and abs(res.fun - 1) < 1e-6
        line = f"{d:5d}d  {int(truth.sum())}/{truth.size} inside: "
        for name in args.methods:
            if (device == "gpu" and name in CPU_ONLY) or (device == "cpu" and name in GPU_ONLY):
                continue
            try:
                with jax.default_device(jax.devices("cuda" if device == "gpu" else "cpu")[0]):
                    _, _, out, _ = run_method(name, d, 4, device, 1, data=data)
                wrong = (np.asarray(out) != truth) & ~ambiguous
                line += f" {name}={'ok' if not wrong.any() else f'{int(wrong.sum())} WRONG'}"
            except Exception as e:  # noqa: BLE001
                line += f" {name}={type(e).__name__}"
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        print(line, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--dims", nargs="+", type=int, default=list(DIMS))
    p.add_argument("--batch", nargs="+", type=int, default=[1, 10, 100], help="1 = unbatched")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--budget", type=float, default=0.6, help="stop growing a method past this many s/repetition")
    p.add_argument("--check", action="store_true", help="correctness against HiGHS instead of speed")
    p.add_argument("--check-max-dim", type=int, default=50)
    p.add_argument("--list", action="store_true")
    p.add_argument("--verbose", action="store_true", help="print tracebacks of failing methods")
    p.add_argument("--mem-fraction", type=float, default=0.4, help="kill a cell above this share of RAM")
    p.add_argument("--cell-timeout", type=float, default=300)
    p.add_argument("--out", help="where the speed table is written (default benchmark/local_results/)")
    p.add_argument("--cell", nargs=3, metavar=("METHOD", "DIM", "BATCH"), help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.list:
        print("\n".join(METHODS))
        return
    os.environ["JAX_PLATFORMS"] = "cuda,cpu" if args.device == "gpu" else "cpu"
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import warnings
    warnings.filterwarnings("ignore")
    import moreau.jax  # noqa: F401 — enables float64, as in the benchmark
    if args.cell:
        cell(args, args.device)
    else:
        (check if args.check else speed)(args, args.device)


if __name__ == "__main__":
    main()
