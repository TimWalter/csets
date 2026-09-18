from typing import Literal

import chex
import jax
import jax.numpy as jnp

from moreau.jax import Cones, Settings
from jaxtyping import Array, Float, Bool, PRNGKeyArray

from . import MoreauSolver, SetType, ContinuousSetType
from .utils import safe_norm

ZonotopeType = SetType("Zonotope")


@chex.dataclass(frozen=True)
class Zonotope:
    r"""
    A zonotope is a convex set defined as
    $Z = \left{x | x = c + Sum_i \beta_i * G_i, \beta_i in [-1, 1] \right}$
    with its centre $c \in \mathbb{R}^d$ and generators $G\in \mathbb{R}^{d\times n}$.

     Attributes:
        centre: The centre of the zonotope.
        generator: The generator matrix of the zonotope.
    """
    centre: Float[Array, "d"]
    generator: Float[Array, "d n"]

    @classmethod
    def random(cls,
               key: PRNGKeyArray,
               dim: int | None = None,
               centre: Float[Array, "d"] | None = None,
               nr_generators: int | None = None,
               distribution: Literal["uniform", "exp", "gamma"] = "uniform"
               ) -> ZonotopeType["d"]:
        r"""
        Generate a random zonotope.

        Args:
            key: PRNG key.
            dim: Dimension of the zonotope.
                 Defaults to `centre`'s length if `centre` is given, otherwise a random integer in [1, 10].
            centre: Center of the zonotope.
                    Defaults to `10 * randn(dim)`.
            nr_generators: Number of generators.
                           Defaults to `2 * dim`.
            distribution: Distribution of generator lengths: "uniform" ~ U(0, 1), "exp" ~ Exp(1), "gamma" ~ Gamma(2, 1).
                          Defaults to "uniform".

        Returns:
            The random zonotope.
        """
        dim_key, center_key, length_key, direction_key = jax.random.split(key, 4)

        if dim is None:
            dim = centre.shape[0] if centre is not None else jax.random.randint(dim_key, (), 1, 11).item()

        if centre is None:
            centre = 10 * jax.random.normal(center_key, (dim,))

        if nr_generators is None:
            nr_generators = 2 * dim

        if distribution == "gamma":
            lengths = jax.random.gamma(length_key, 2.0, (nr_generators,))
        elif distribution == "exp":
            lengths = jax.random.exponential(length_key, (nr_generators,))
        else:
            lengths = jax.random.uniform(length_key, (nr_generators,))

        directions = jax.random.normal(direction_key, (dim, nr_generators))
        directions = directions / safe_norm(directions, axis=0, keepdims=True)
        generator = directions * lengths

        return cls(centre=centre, generator=generator)

    def sample(self: ZonotopeType["d"],
               key: PRNGKeyArray,
               num_samples: int,
               ) -> Float[Array, "{num_samples} d"]:
        r"""
        Sample from the zonotope cheaply but not uniformly.

        Args:
            key: PRNG key.
            num_samples: The number of samples to draw.

        Returns:
            Sampled points.
        """
        factors = jax.random.uniform(key, (num_samples, self.generator.shape[1]), minval=-1, maxval=1)

        return self.centre[None, :] + factors @ self.generator.T

    def support(self: ZonotopeType["d"],
                direction: Float[Array, "d"]
                ) -> Float[Array, ""]:
        r"""
        Compute the support of the zonotope in the given direction.

        Args:
            direction: The direction in which to compute the support, expected to be of unit length.

        Returns:
            Support in the given direction.
        """
        return direction @ self.centre + jnp.sum(direction @ self.generator, axis=-1)

    def interval(self: ZonotopeType["d"]
                 ) -> ZonotopeType["d"]:  # TODO should return an actual interval object later
        r"""
        Return the over-approximative interval in zonotope representation.

        Returns:
            Interval in zonotope representation.
        """
        return Zonotope(centre=self.centre, generator=jnp.diag(jnp.linalg.norm(self.generator, ord=1, axis=-1)))

    def __rmatmul__(self: ZonotopeType["d"],
                    transform: Float[Array, "m d"]
                    ) -> ZonotopeType["m"]:
        r"""
        Apply a linear map to the zonotope from the left: A @ Z.

        Args:
            transform: The linear transformation matrix.

        Returns:
            The transformed zonotope.
        """
        return Zonotope(
            centre=transform @ self.centre,
            generator=transform @ self.generator
        )

    def minkowski_sum(self: ZonotopeType["d"],
                      other: ZonotopeType["d"]
                      ) -> ZonotopeType["d"]:
        """
        Compute the Minkowski sum of two zonotopes.

        Args:
            other: The other zonotope.

        Returns:
            Sumset of the two zonotopes.

        Notes:
            Could also override the __add__ operator.
        """
        return Zonotope(
            centre=self.centre + other.centre,
            generator=jnp.concat([self.generator, other.generator], axis=-1)
        )

    def make_contains(self: ZonotopeType["d"],
                      inner: ContinuousSetType["d"] | Float[Array, "d"],
                      ) -> MoreauSolver:
        r"""
        Set up the optimisation problem for the containment check, reusable across calls as long as the
        shapes and type stay the same.

        Args:
            inner: An example of the kind of continuous set or point, whose containment to check.

        Returns:
            A moreau solver; consume it through `contains`.
        """
        if isinstance(inner, Zonotope):
            return self._make_contains_zonotope(inner)
        elif isinstance(inner, Array):
            return self._make_contains_point()
        else:
            raise TypeError(f"Unsupported type for inner: {type(inner)}")

    def contains(self: ZonotopeType["d"],
                 inner: ContinuousSetType["d"] | Float[Array, "d"],
                 solver: MoreauSolver
                 ) -> Bool[Array, ""]:
        """
        Checks if a continuous set or point is contained in the zonotope.

        The solver must have been constructed via `make_contains` with an `inner` of the same type
        and shape, and with a zonotope of the same shape as `self`.

        Args:
            inner: The continuous set or point to check.
            solver: Pre-compiled moreau solver.

        Returns:
            Flag indicating containment.

        Notes:
            Could also override the __contains__ operator.
        """
        if isinstance(inner, Zonotope):
            return self._contains_zonotope(inner, solver)
        elif isinstance(inner, Array):
            return self._contains_point(inner, solver)
        else:
            raise TypeError(f"Unsupported type for inner: {type(inner)}")

    def _make_contains_point(self: ZonotopeType["d"]) -> MoreauSolver:
        r"""
        Build a solver for the point containment problem of a zonotope, namely
        $1\geq\min_{\beta\in\mathbb{R}^n} \norm{\beta}_\infty\,, \text{s.t.} point=c+G\beta.
        We canonicalise to:
            min_z q^T z
            s.t. A z + s = b, s \in K

        with z=[\beta, t], q=[\bm{0}, 1], A=[[G, \bm{0}],
                                             [I, -\bm{1}],
                                             [-I, -\bm{1}]],
        b=[p - c, \bm{0}, \bm{0}], and K a zero cone of dimension d followed by a non-negative cone.
        See Kulmburg, A., Althoff, M. (2021): "On the co-NP-Completeness of the Zonotope Containment Problem", Eq. (6).

        Returns:
            A moreau solver; consume it through `contains`.
        """
        d, n = self.generator.shape

        P_row_offsets = jnp.zeros(n + 2, dtype=jnp.int32)
        P_col_indices = jnp.array([], dtype=jnp.int32)

        A_row_offsets = jnp.concat([jnp.arange(0, d * n + 1, n), d * n + jnp.arange(2, 4 * n + 1, 2)])
        bound_cols = jnp.stack([jnp.arange(n), jnp.full(n, n)], axis=1).flatten()
        A_col_indices = jnp.concat([jnp.tile(jnp.arange(n), d), bound_cols, bound_cols])

        cones = Cones(num_zero_cones=d, num_nonneg_cones=2 * n)

        return MoreauSolver(n=n + 1, m=d + 2 * n,
                            P_row_offsets=P_row_offsets, P_col_indices=P_col_indices,
                            A_row_offsets=A_row_offsets, A_col_indices=A_col_indices,
                            cones=cones,
                            settings=Settings(solver='active_set'))

    def _contains_point(self: ZonotopeType["d"],
                        point: Float[Array, "d"],
                        solver: MoreauSolver
                        ) -> Bool[Array, ""]:
        """
        Check whether a point is contained in the zonotope, up to the solver tolerance.

        Args:
            point: The point to check.
            solver: Pre-compiled moreau solver, constructed via `_make_contains_point`.

        Returns:
            Flag indicating containment.
        """
        n = self.generator.shape[1]

        p = jnp.array([])
        a = jnp.concat([self.generator.flatten(),
                        jnp.tile(jnp.array([1.0, -1.0]), n),
                        jnp.full(2 * n, -1.0)])
        q = jnp.concat([jnp.zeros(n), jnp.ones(1)])
        b = jnp.concat([point - self.centre, jnp.zeros(2 * n)])

        return solver.solve(p, a, q, b).x[n] <= 1.0 + 1e-6

    def _make_contains_zonotope(self: ZonotopeType["d"],
                                inner: ZonotopeType["d"]
                                ) -> MoreauSolver:
        r"""
        Build a solver for the zonotope containment problem of a zonotope, namely
        $$
        1\geq\min_{\beta\in\mathbb{R}^n_s, \Gamma\in\mathbb{R}^{n_s\times n_i}} \norm{[\beta, \Gamma]}_\infty
        \text{s.t.} G_i=G_s\Gamma
        c_s-c_i=G_s\beta
        $$
        We canonicalise the row sums through auxiliary variables U >= |\Gamma| and V >= |\beta| to:
            min_z q^T z
            s.t. A z + s = b, s \in K

        with z=[\Gamma, \beta, U, V, t], q=[\bm{0}, 1], K a zero cone of dimension d (m + 1)
        followed by a non-negative cone, and the rows of A grouped as
            d m rows: G_1 \Gamma = G_2                     (zero cone)
            d rows:   G_1 \beta = c_1 - c_2                (zero cone)
            2 n m rows: +-\Gamma - U <= 0                  (non-negative cone)
            2 n rows:   +-\beta - V <= 0                   (non-negative cone)
            n rows:     \sum_j U_kj + V_k - t <= 0         (non-negative cone)

        The encoding is sufficient but not necessary: exact zonotope containment is co-NP-complete, so a contained pair
        may still be reported as not contained.

        See Sadraddini, S., Tedrake, R. (2019): "Linear Encodings for Polytope Containment Problems", Eq. (5)

        Returns:
            A moreau solver; consume it through `contains`.
        """
        d, n = self.generator.shape
        m = inner.generator.shape[1]

        num_variables = 2 * n * m + 2 * n + 1
        num_zero_cones = d * (m + 1)
        num_nonneg_cones = 2 * n * m + 3 * n

        P_row_offsets = jnp.zeros(num_variables + 1, dtype=jnp.int32)
        P_col_indices = jnp.array([], dtype=jnp.int32)

        row_sizes = jnp.concat([jnp.full(num_zero_cones, n, dtype=jnp.int32),
                                jnp.full(2 * n * m + 2 * n, 2, dtype=jnp.int32),
                                jnp.full(n, m + 2, dtype=jnp.int32)])
        A_row_offsets = jnp.concat([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(row_sizes)])

        mapping_cols = jnp.arange(n * m)
        weight_cols = n * m + jnp.arange(n)
        abs_mapping_cols = n * m + n + mapping_cols
        abs_weight_cols = 2 * n * m + n + jnp.arange(n)
        t_col = 2 * n * m + 2 * n

        shape_cols = jnp.tile((jnp.arange(m)[:, None] + m * jnp.arange(n)[None, :]).flatten(), d)
        centre_cols = jnp.tile(weight_cols, d)
        mapping_bound_cols = jnp.stack([mapping_cols, abs_mapping_cols], axis=1).flatten()
        weight_bound_cols = jnp.stack([weight_cols, abs_weight_cols], axis=1).flatten()
        row_sum_cols = jnp.concat([abs_mapping_cols.reshape(n, m),
                                   abs_weight_cols[:, None],
                                   jnp.full((n, 1), t_col)], axis=1).flatten()

        A_col_indices = jnp.concat([shape_cols, centre_cols,
                                    mapping_bound_cols, mapping_bound_cols,
                                    weight_bound_cols, weight_bound_cols,
                                    row_sum_cols])

        cones = Cones(num_zero_cones=num_zero_cones, num_nonneg_cones=num_nonneg_cones)

        return MoreauSolver(n=num_variables, m=num_zero_cones + num_nonneg_cones,
                            P_row_offsets=P_row_offsets, P_col_indices=P_col_indices,
                            A_row_offsets=A_row_offsets, A_col_indices=A_col_indices,
                            cones=cones,
                            settings=Settings(solver='active_set'))

    def _contains_zonotope(self: ZonotopeType["d"],
                           inner: ZonotopeType["d"],
                           solver: MoreauSolver
                           ) -> Bool[Array, ""]:
        """
        Check whether another zonotope is contained in this one, up to the solver tolerance.

        Args:
            inner: The zonotope to check.
            solver: Pre-compiled moreau solver, constructed via `_make_contains_zonotope`.

        Returns:
            Flag indicating containment.
        """
        n = self.generator.shape[1]
        m = inner.generator.shape[1]

        p = jnp.array([])
        a = jnp.concat([jnp.repeat(self.generator, m, axis=0).flatten(),
                        self.generator.flatten(),
                        jnp.tile(jnp.array([1.0, -1.0]), n * m),
                        jnp.full(2 * n * m, -1.0),
                        jnp.tile(jnp.array([1.0, -1.0]), n),
                        jnp.full(2 * n, -1.0),
                        jnp.tile(jnp.concat([jnp.ones(m + 1), jnp.array([-1.0])]), n)])
        q = jnp.zeros(2 * n * m + 2 * n + 1).at[-1].set(1.0)
        b = jnp.concat([inner.generator.flatten(),
                        self.centre - inner.centre,
                        jnp.zeros(2 * n * m + 3 * n)])

        return solver.solve(p, a, q, b).x[-1] <= 1.0 + 1e-6
