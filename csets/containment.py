r"""
Exact point containment for zonotopes without solving an LP per point.

A point $p$ lies in $Z = \{c + G\beta \mid \|\beta\|_\infty \leq 1\}$ iff the affine set
$A = \{\beta \mid G\beta = r\}$, $r = p - c$, meets the box $[-1, 1]^m$, i.e. iff the gauge
$t^*(r) = \min\{\|\beta\|_\infty \mid G\beta = r\}$ is at most one. All three paths below answer
exactly that question, with the same tolerance: inside iff $t^* \leq 1 + $ `TOL`.

- Facets, while the zonotope has few of them ($\binom{m}{d-1} \leq$ `MAX_FACETS`): with the facet
  normals $h$, $t^* = \max_h |h^\top r| / \sum_j |h^\top g_j|$ in closed form.
- Certificates, otherwise: alternating projections between $A$ and a shrunk box look for a proof
  either way. A $\beta \in A$ with $\|\beta\|_\infty \leq 1 + $ `TOL` proves "inside"; a direction
  $y$ with $y^\top r > (1 + $ `TOL`$) \|G^\top y\|_1$ proves "outside" (it separates $p$ from $Z$).
  By LP duality one of the two always exists; both are checked against $G$ itself.
- LP, for whatever neither settles (points near the boundary, flat or ill-conditioned $G$):
  Kulmburg, A., Althoff, M. (2021): "On the co-NP-Completeness of the Zonotope Containment
  Problem", Eq. (6), on Moreau's interior point method.

Which of the first two applies depends on the shapes alone and is fixed in `PointContainment`.
Whether the LP runs depends on the data: it runs, for all points of the call, only when some
point is left unsettled. That is a `lax.cond`, which `jax.vmap` would turn into a `select` that
always runs the LP; `PointContainment` therefore batches itself (`custom_vmap`), folding every
vmapped axis into one leading batch so the whole call keeps a single `cond`.

The answer is a boolean, so nothing here is differentiable; the inputs are cut from the
gradient, and differentiating a function that uses the answer still works.
"""
import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np

from jaxtyping import Array, Float, Bool

from . import MoreauSolver

TOL = 1e-6  # inside iff the gauge is at most 1 + TOL, on every path
MAX_FACETS = 300  # facet normals up to which the facets are cheaper than the certificates
SHRINK = 0.9  # half-width of the box the certificates project onto; see `_by_certificates`
MAX_ITER = 60  # projections before a point is left to the LP
CHECK_EVERY = 4  # projections between checks of the certificates


class PointContainment:
    """
    Point containment for zonotopes of one shape; built by `Zonotope.make_contains` and consumed
    by `Zonotope.contains`. Holds what depends on the shape alone: which fast path applies, the
    facet index tables, and the Moreau solver of the LP fallback.
    """

    def __init__(self, dim: int, nr_generators: int):
        """
        Args:
            dim: Dimension d of the zonotopes.
            nr_generators: Number of generators m of the zonotopes.
        """
        d, m = dim, nr_generators
        self.dim, self.nr_generators = d, m
        if m < d:  # never full-dimensional: neither facets nor certificates, only the LP
            self.method = "lp"
        elif d == 1 or math.comb(m, d - 1) <= MAX_FACETS:
            self.method = "facets"
            # Every (d-1)-subset of the generators, and for each coordinate the other d-1 rows:
            # the minors of the generalised cross product that is the subset's facet normal.
            self._subsets = np.array(list(itertools.combinations(range(m), d - 1)), dtype=np.int32)
            self._rows = np.array([[j for j in range(d) if j != i] for i in range(d)], dtype=np.int32)
            self._signs = np.array([(-1.0) ** i for i in range(d)])
        else:
            self.method = "certificates"
        self.lp = _make_lp(d, m)
        self._decide = self._make_decide()

    def __call__(self,
                 centre: Float[Array, "d"],
                 generator: Float[Array, "d m"],
                 points: Float[Array, "n d"]
                 ) -> Bool[Array, "n"]:
        """
        Whether each point lies in the zonotope.

        Args:
            centre: Centre of the zonotope.
            generator: Generator matrix of the zonotope.
            points: The points to check.

        Returns:
            Flag per point.
        """
        centre, generator, points = jax.lax.stop_gradient((centre, generator, points))
        return self._decide(centre[None], generator[None], points[None])[0]

    def _make_decide(self):
        """The whole decision over an explicit leading batch of sets, batching itself under vmap."""

        @jax.custom_batching.custom_vmap
        def decide(C, G, P):
            inside, settled = self._fast(C, G, P)
            return jax.lax.cond(settled.all(),
                                lambda: inside,
                                lambda: jnp.where(settled, inside, _by_lp(self.lp, C, G, P)))

        @decide.def_vmap
        def decide_batched(axis_size, in_batched, C, G, P):
            # Fold the vmapped axis into the leading batch and decide once more: one cond for all.
            C, G, P = (x if batched else jnp.broadcast_to(x, (axis_size, *x.shape))
                       for x, batched in zip((C, G, P), in_batched))
            K = C.shape[1]
            out = decide(C.reshape(axis_size * K, *C.shape[2:]),
                         G.reshape(axis_size * K, *G.shape[2:]),
                         P.reshape(axis_size * K, *P.shape[2:]))
            return out.reshape(axis_size, K, -1), True

        return decide

    def _fast(self, C, G, P):
        """(inside, settled) per point, without the LP."""
        if self.method == "facets":
            return _by_facets(C, G, P, self._subsets, self._rows, self._signs)
        if self.method == "certificates":
            return _by_certificates(C, G, P)
        unsettled = jnp.zeros(P.shape[:2], dtype=bool)
        return unsettled, unsettled


def _by_facets(C: Float[Array, "K d"],
               G: Float[Array, "K d m"],
               P: Float[Array, "K N d"],
               subsets, rows, signs
               ) -> tuple[Bool[Array, "K N"], Bool[Array, "K N"]]:
    r"""
    (inside, settled): a full-dimensional zonotope is the intersection of the slabs
    $|h^\top(x - c)| \leq \sum_j |h^\top g_j|$, one per $d-1$ generators whose normal $h$ (their
    generalised cross product) is nonzero.

    Only for a full-dimensional zonotope; a flat one is left unsettled. Flat means either no
    normal survives rounding, or some normal is orthogonal to every generator (rank $d-1$: the
    normal of the hull, whose slab only says "in the hull"). The thresholds are absolute, in
    units of the generators, since relative to each other rounding noise would pass for normals.
    """
    d = G.shape[-2]
    r = P - C[:, None, :]
    if d == 1:
        reach = jnp.abs(G).sum((-2, -1))                                    # (K,)
        inside = jnp.abs(r[..., 0]) <= (1 + TOL) * reach[:, None]
        return inside, jnp.broadcast_to((reach > 0)[:, None], inside.shape)
    minors = jnp.moveaxis(G[..., subsets], -3, -2)[..., rows, :]            # (K, F, d, d-1, d-1)
    H = jnp.linalg.det(minors) * signs                                      # (K, F, d)
    reach = jnp.abs(H @ G).sum(-1)                                          # (K, F)
    unit = jnp.abs(G).max((-2, -1))[:, None]                                # (K, 1)
    scale = jnp.abs(H).max(-1)                                              # |h| <= (d-1)! unit^(d-1)
    valid = scale > 1e-9 * unit ** (d - 1)                                  # degenerate subsets drop out
    spans = reach > 1e-9 * scale * unit                                     # h not orthogonal to all of G
    full = valid.any(-1) & (spans | ~valid).all(-1)
    within = jnp.abs(jnp.einsum("knd,kfd->knf", r, H)) <= (1 + TOL) * reach[:, None, :]
    inside = (within | ~valid[:, None, :]).all(-1)
    return inside, jnp.broadcast_to(full[:, None], inside.shape)


def _by_certificates(C: Float[Array, "K d"],
                     G: Float[Array, "K d m"],
                     P: Float[Array, "K N d"]
                     ) -> tuple[Bool[Array, "K N"], Bool[Array, "K N"]]:
    r"""
    (inside, settled) from alternating projections between $A = \{\beta \mid G\beta = r\}$ and the
    shrunk box $[-s, s]^m$, $s = $ `SHRINK`.

    If $p$ lies in $c + s(Z - c)$, the two sets intersect and the projections converge into the
    intersection, so an iterate soon has $\|\beta\|_\infty \leq s < 1$; the full box would only be
    approached asymptotically. If $p$ lies outside $Z$, the gap $u$ between the iterates in $A$ and
    in the box yields $y = (GG^\top)^{-1}Gu$. Points between (and slow ones) stay unsettled.

    The projection onto $A$ is $x - G^\top(GG^\top)^{-1}(Gx - r)$ with a Cholesky factor of
    $GG^\top$, shared by all points of a set. No orthonormal basis is formed: on XLA:CPU a QR of
    $G^\top$ costs seconds at 1000d, and a triangular solve against all of $G$ milliseconds at 50d,
    whereas solves with the points as right-hand sides take microseconds. A flat $G$ fails the
    factorisation; its NaNs fail both checks and leave the points to the LP.
    """
    Gt = jnp.swapaxes(G, -1, -2)                                            # (K, m, d)
    factor = (jnp.linalg.cholesky(G @ Gt), True)

    def gram_solve(v):                                                      # (GG')^-1 v for v (K, N, d)
        return jnp.swapaxes(jax.scipy.linalg.cho_solve(factor, jnp.swapaxes(v, -1, -2)), -1, -2)

    r = P - C[:, None, :]                                                   # (K, N, d)
    rscale = 1 + jnp.abs(r).max(-1)
    eps = 1000 * jnp.finfo(P.dtype).eps

    def to_affine(x):
        return x - gram_solve(x @ Gt - r) @ G

    def certify(x):
        in_affine = jnp.abs(x @ Gt - r).max(-1) <= 1e-9 * rscale
        inside = in_affine & (jnp.abs(x).max(-1) <= 1 + TOL)
        y = gram_solve((x - jnp.clip(x, -SHRINK, SHRINK)) @ Gt)             # (K, N, d)
        reach = jnp.abs(y @ G).sum(-1)                                      # ||G'y||_1
        support = (y * r).sum(-1)
        outside = support - (1 + TOL) * reach > eps * (reach + jnp.abs(y * r).sum(-1))
        return inside, outside & ~inside

    def unsettled(state):
        _, inside, outside, it = state
        return (it < MAX_ITER) & ~(inside | outside).all()

    def step(state):
        x, inside, outside, it = state
        x = jax.lax.fori_loop(0, CHECK_EVERY, lambda _, v: to_affine(jnp.clip(v, -SHRINK, SHRINK)), x)
        now_inside, now_outside = certify(x)
        return x, inside | now_inside, outside | now_outside, it + CHECK_EVERY

    x = gram_solve(r) @ G                                                   # least-norm solutions
    _, inside, outside, _ = jax.lax.while_loop(unsettled, step, (x, *certify(x), 0))
    return inside, inside | outside


def _make_lp(d: int, m: int) -> MoreauSolver:
    r"""
    The Moreau solver for the point containment LP
    $1\geq\min_{\beta\in\mathbb{R}^m} \norm{\beta}_\infty\,, \text{s.t.} p=c+G\beta$, canonicalised to
        min_z q^T z
        s.t. A z + s = b, s \in K

    with z=[\beta, t], q=[\bm{0}, 1], A=[[G, \bm{0}],
                                         [I, -\bm{1}],
                                         [-I, -\bm{1}]],
    b=[p - c, \bm{0}, \bm{0}], and K a zero cone of dimension d followed by a non-negative cone.
    See Kulmburg, A., Althoff, M. (2021): "On the co-NP-Completeness of the Zonotope Containment Problem", Eq. (6).

    Interior point: the active set method answers wrongly from about 50 dimensions on.
    """
    from moreau.jax import Cones, Settings

    P_row_offsets = jnp.zeros(m + 2, dtype=jnp.int32)
    P_col_indices = jnp.array([], dtype=jnp.int32)

    A_row_offsets = jnp.concat([jnp.arange(0, d * m + 1, m), d * m + jnp.arange(2, 4 * m + 1, 2)])
    bound_cols = jnp.stack([jnp.arange(m), jnp.full(m, m)], axis=1).flatten()
    A_col_indices = jnp.concat([jnp.tile(jnp.arange(m), d), bound_cols, bound_cols])

    return MoreauSolver(n=m + 1, m=d + 2 * m,
                        P_row_offsets=P_row_offsets, P_col_indices=P_col_indices,
                        A_row_offsets=A_row_offsets, A_col_indices=A_col_indices,
                        cones=Cones(num_zero_cones=d, num_nonneg_cones=2 * m),
                        settings=Settings(solver="ipm"),
                        enable_grad=False)


def _by_lp(solver: MoreauSolver,
           C: Float[Array, "K d"],
           G: Float[Array, "K d m"],
           P: Float[Array, "K N d"]
           ) -> Bool[Array, "K N"]:
    """
    Inside per point from the LP, accepted only with a solution that is itself a certificate.

    One set after another (`lax.map`), its points as one batch of LPs: every LP carries its own
    copy of G, so batching all sets at once would hold K N d m values, 16 GB at 1000d with
    100 sets of 10 points, and XLA reserves that even while the fallback never runs.
    """
    N, d = P.shape[1:]
    m = G.shape[-1]

    def one(g, r):
        a = jnp.concat([g.flatten(), jnp.tile(jnp.array([1.0, -1.0]), m), jnp.full(2 * m, -1.0)])
        q = jnp.concat([jnp.zeros(m), jnp.ones(1)])
        b = jnp.concat([r, jnp.zeros(2 * m)])
        beta = solver.solve(jnp.array([]), a, q, b).x[:m]
        in_affine = jnp.abs(g @ beta - r).max() <= TOL * (1 + jnp.abs(r).max())
        return in_affine & (jnp.abs(beta).max() <= 1 + TOL)

    def per_set(set_):
        c, g, p = set_
        return jax.vmap(one, in_axes=(None, 0))(g, p - c)

    return jax.lax.map(per_set, (C, G, P))
