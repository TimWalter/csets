import math
import itertools
from typing import Literal

import jax
import cvxpy as cp
import jax.numpy as jnp

from jaxtyping import Array, PRNGKeyArray, Float, Bool

from .types import ContinuousSetType, ZonotopeType, IntervalType, PolytopeType
from .settings import config
from .solver import Solver
from .utils import safe_norm, pytree_dataclass, generalised_cross



@pytree_dataclass
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
        return direction @ self.centre + jnp.abs(direction @ self.generator).sum()

    def interval(self: ZonotopeType["d"]
                 ) -> IntervalType["d"]:
        r"""
        Compute the interval hull, the smallest interval containing the zonotope.

        Returns:
            The interval with the zonotope's centre and radius $\sum_j |g_j|$.
        """
        from .interval import Interval

        return Interval(centre=self.centre, radius=jnp.abs(self.generator).sum(-1))

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
                      ) -> Solver | None:
        r"""
        Set up the optimisation problem for the containment check, reusable across calls as long as the
        shapes and type stay the same.

        Args:
            inner: An example of the kind of continuous set or point, whose containment to check.

        Returns:
            A moreau solver, or None if the check needs none; consume it through `contains`.
        """
        if isinstance(inner, Zonotope):
            return self._make_contains_zonotope(inner)
        elif isinstance(inner, Array):
            return self._make_contains_point()
        else:
            raise TypeError(f"Unsupported type for inner: {type(inner)}")

    def contains(self: ZonotopeType["d"],
                 inner: ContinuousSetType["d"] | Float[Array, "d"],
                 solver: Solver | None
                 ) -> Bool[Array, ""]:
        """
        Checks if a continuous set or point is contained in the zonotope.

        Args:
            inner: The continuous set or point to check.
            solver: Pre-compiled solver, which must have been constructed via `make_contains` with an `inner`
            of the same type and shape, and with a zonotope of the same shape as `self`.

        Returns:
            Flag indicating containment.

        Notes:
            Could also override the __contains__ operator.
        """
        outer, inner = jax.lax.stop_gradient((self, inner))
        if isinstance(inner, Zonotope):
            _, bound, residuals = solver(inner.centre, inner.generator, outer.centre, outer.generator)
        elif isinstance(inner, Array) and solver is None:
            return outer.polytope().contains(inner)
        elif isinstance(inner, Array):
            _, bound, residuals = solver(inner, outer.centre, outer.generator)
        else:
            raise TypeError(f"Unsupported type for inner: {type(inner)}")
        return (residuals <= config.tolerance).all() & (bound <= 1 + config.tolerance)

    def _make_contains_point(self: ZonotopeType["d"]) -> Solver | None:
        r"""
        Build the solver for the point containment LP, namely
        $$
        1\geq\min_{\beta\in\mathbb{R}^n} \norm{\beta}_\infty\,, \text{s.t.} p=c+G\beta\,.
        $$
        See Kulmburg, A., Althoff, M. (2021): "On the co-NP-Completeness of the Zonotope Containment Problem", Eq. (6).

        While the zonotope has few facets ($\binom{n}{d-1} \leq 300$), its halfspace representation decides
        instead, and no solver is needed; this assumes the zonotope is full-dimensional. Otherwise the
        alternating projections answer most points without the LP.

        Returns:
            A solver, called with (p, c, G), or None if the halfspace representation decides; consume it through
            `contains`.
        """
        d, n = self.generator.shape
        if n >= d and math.comb(n, d - 1) <= 300:
            return None
        parameters = [
            point := cp.Parameter(d),
            centre := cp.Parameter(d),
            generator := cp.Parameter((d, n))
        ]
        variables = [
            weights := cp.Variable(n)
        ]
        objective = cp.Minimize(cp.norm(weights, "inf"))
        constraints = [
            point == centre + generator @ weights
        ]
        problem = cp.Problem(objective, constraints)
        heuristic = _alternating_projections if n >= d else None
        return Solver(problem, parameters, variables, heuristic,
                      solver="MOREAU",
                      solver_args={"solver": "ipm",
                                   "device": jnp.empty(0).device.platform.replace("gpu", "cuda"),
                                   "enable_grad": False})

    def _make_contains_zonotope(self: ZonotopeType["d"],
                                inner: ZonotopeType["d"]
                                ) -> Solver:
        r"""
        Build the solver for the zonotope containment problem of a zonotope, namely
        $$
        1\geq\min_{\beta\in\mathbb{R}^n_o, \Gamma\in\mathbb{R}^{n_o\times n_i}} \norm{[\Gamma, \beta]}_\infty
        \text{s.t.} G_i=G_o\Gamma
        c_o-c_i=G_o\beta\,.
        $$
        See Sadraddini, S., Tedrake, R. (2019): "Linear Encodings for Polytope Containment Problems", Eq. (5)

        Returns:
            A solver, called with (c_i, G_i, c_o, G_o); consume it through `contains`.

        Notes:
            The condition is sufficient but not necessary, so a contained pair may still be reported as not contained.
        """
        d, outer_n = self.generator.shape
        inner_n = inner.generator.shape[1]
        parameters = [
            inner_centre := cp.Parameter(d),
            inner_generator := cp.Parameter((d, inner_n)),
            outer_centre := cp.Parameter(d),
            outer_generator := cp.Parameter((d, outer_n))
        ]
        variables = [
            weights := cp.Variable(outer_n),
            mapping := cp.Variable((outer_n, inner_n))
        ]
        objective = cp.Minimize(cp.norm(cp.hstack([mapping, weights[:, None]]), "inf"))
        constraints = [inner_generator == outer_generator @ mapping,
                       outer_centre - inner_centre == outer_generator @ weights]
        problem = cp.Problem(objective, constraints)
        return Solver(problem, parameters, variables,
                      solver="MOREAU",
                      solver_args={"solver": "ipm",
                                   "device": jnp.empty(0).device.platform.replace("gpu", "cuda"),
                                   "enable_grad": False})


    def polytope(self: ZonotopeType["d"]) -> PolytopeType:
        r"""
        Convert the zonotope to a polytope in halfspace representation.

        Returns:
            The polytope, with $2\binom{n}{d-1}$ halfspaces.
        """
        d, n = self.generator.shape
        subsets = jnp.array(list(itertools.combinations(range(n), d - 1)), dtype=jnp.int32)
        halfspace = jax.vmap(generalised_cross)(self.generator[:, subsets].transpose(1, 0, 2))

        length = jnp.linalg.norm(halfspace, axis=-1, keepdims=True)
        valid = length > 1e-9 * jnp.abs(self.generator).max() ** (d - 1)
        normal = jnp.where(valid, halfspace / jnp.where(valid, length, 1), 0)
        reach = jnp.abs(normal @ self.generator).sum(-1)
        offset = normal @ self.centre

        from .polytope import Polytope

        return Polytope(normal=jnp.concatenate([normal, -normal]),
                        anchor=jnp.concatenate([offset + reach, reach - offset]))


def _alternating_projections(point: Float[Array, "d"],
                             centre: Float[Array, "d"],
                             generator: Float[Array, "d m"]
                             ) -> tuple[Bool[Array, ""], tuple[Float[Array, "m"]]]:
    r"""
    Check if a point inside the zonotope by using the fact that the point is inside iff the affine subspace
    $A = \{\beta \mid G\beta = r\}$, $r = p - c$, intersects the box $[-1, 1]^m$.
    The projections alternate between $A$ and the shrunk box $[-s, s]^m$, $s = 0.9$.

    Args:
        point: The point.
        centre: The zonotope's centre.
        generator: The zonotope's generators.

    Returns:
        Flag indicating whether the solve was successful and the solution.
    """
    shrink = 0.9
    G = generator
    threshold = 1 + config.tolerance
    factor = (jnp.linalg.cholesky(G @ G.T), True)
    r = point - centre

    def to_affine(x):
        return x - G.T @ jax.scipy.linalg.cho_solve(factor, G @ x - r)

    def project(x, times):
        return jax.lax.fori_loop(0, times, lambda _, v: to_affine(jnp.clip(v, -shrink, shrink)), x)

    def settles(x):
        in_affine = jnp.abs(G @ x - r).max() <= 1e-9 * (1 + jnp.abs(r).max())
        inside = jnp.abs(x).max() <= threshold
        y = jax.scipy.linalg.cho_solve(factor, G @ (x - jnp.clip(x, -shrink, shrink)))
        reach = jnp.abs(G.T @ y).sum()  # ||G'y||_1, the support of Z - c along y
        rounding = 1000 * jnp.finfo(r.dtype).eps * (reach + jnp.abs(y * r).sum())
        outside = y @ r - threshold * reach > rounding
        return in_affine & (inside | outside)

    def unsettled(state):
        _, settled, projections = state
        return (projections < 64) & ~settled

    def step(state):
        x, _, projections = state
        x = project(x, 4)
        return x, settles(x), projections + 4

    x = project(G.T @ jax.scipy.linalg.cho_solve(factor, r), 16)
    x, settled, _ = jax.lax.while_loop(unsettled, step, (x, settles(x), 16))
    return settled, (x,)
