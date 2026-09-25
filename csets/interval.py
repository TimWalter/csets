import jax
import jax.numpy as jnp

from jaxtyping import Array, PRNGKeyArray, Float, Bool

from .types import ContinuousSetType, IntervalType, ZonotopeType
from .settings import config
from .utils import pytree_dataclass



@pytree_dataclass
class Interval:
    r"""
    An interval, or axis-aligned box, is the convex set
    $I = \left\{x \mid c - r \leq x \leq c + r \right\}$
    with its centre $c \in \mathbb{R}^d$ and radius $r \in \mathbb{R}^d_{\geq 0}$.

    Attributes:
        centre: The centre of the interval.
        radius: The radius of the interval along each axis.
    """
    centre: Float[Array, "d"]
    radius: Float[Array, "d"]

    @classmethod
    def random(cls,
               key: PRNGKeyArray,
               dim: int | None = None,
               centre: Float[Array, "d"] | None = None,
               ) -> IntervalType["d"]:
        r"""
        Generate a random interval, distributed as CORA's `interval.generateRandom`.

        Args:
            key: PRNG key.
            dim: Dimension of the interval.
                 Defaults to `centre`'s length if `centre` is given, otherwise a random integer in [1, 10].
            centre: Centre of the interval.
                    Defaults to $U[-2, 2]^d$.

        Returns:
            The random interval, with radius $r_i = R_i u_i / 2$, where $R_i \sim U[0, 10]$ and $u_i \sim U[0, 1]$.
        """
        dim_key, centre_key, range_key, fraction_key = jax.random.split(key, 4)

        if dim is None:
            dim = centre.shape[0] if centre is not None else jax.random.randint(dim_key, (), 1, 11).item()

        if centre is None:
            centre = jax.random.uniform(centre_key, (dim,), minval=-2, maxval=2)

        radius = jax.random.uniform(range_key, (dim,), maxval=10) * jax.random.uniform(fraction_key, (dim,)) / 2

        return cls(centre=centre, radius=radius)

    def sample(self: IntervalType["d"],
               key: PRNGKeyArray,
               num_samples: int,
               ) -> Float[Array, "{num_samples} d"]:
        r"""
        Sample uniformly from the interval.

        Args:
            key: PRNG key.
            num_samples: The number of samples to draw.

        Returns:
            Sampled points.
        """
        factors = jax.random.uniform(key, (num_samples, self.centre.shape[0]), minval=-1, maxval=1)

        return self.centre[None, :] + factors * self.radius[None, :]

    def support(self: IntervalType["d"],
                direction: Float[Array, "d"]
                ) -> Float[Array, ""]:
        r"""
        Compute the support of the interval in the given direction, $d^\top c + |d|^\top r$.

        Args:
            direction: The direction in which to compute the support, expected to be of unit length.

        Returns:
            Support in the given direction.
        """
        return direction @ self.centre + jnp.abs(direction) @ self.radius

    def zonotope(self: IntervalType["d"]) -> ZonotopeType["d"]:
        r"""
        Convert the interval to a zonotope, with one generator per axis.

        Returns:
            The zonotope with the interval's centre and the generators $\operatorname{diag}(r)$.
        """
        from .zonotope import Zonotope

        return Zonotope(centre=self.centre, generator=jnp.diag(self.radius))

    def __rmatmul__(self: IntervalType["d"],
                    transform: Float[Array, "m d"]
                    ) -> IntervalType["m"]:
        r"""
        Apply a linear map to the interval from the left: A @ I.

        The image of an interval is a zonotope in general; this is its interval hull, the smallest interval
        containing it.

        Args:
            transform: The linear transformation matrix.

        Returns:
            The interval with centre $Ac$ and radius $|A| r$.
        """
        return Interval(centre=transform @ self.centre, radius=jnp.abs(transform) @ self.radius)

    def minkowski_sum(self: IntervalType["d"],
                      other: IntervalType["d"]
                      ) -> IntervalType["d"]:
        """
        Compute the Minkowski sum of two intervals.

        Args:
            other: The other interval.

        Returns:
            Sumset of the two intervals.

        Notes:
            Could also override the __add__ operator.
        """
        return Interval(centre=self.centre + other.centre, radius=self.radius + other.radius)

    def make_contains(self: IntervalType["d"],
                      inner: ContinuousSetType["d"] | Float[Array, "d"],
                      ) -> None:
        r"""
        Set up the containment check: nothing to set up, since every check is closed-form.

        Args:
            inner: An example of the kind of continuous set or point, whose containment to check.

        Returns:
            None; pass it to `contains`.
        """
        from .zonotope import Zonotope

        if not isinstance(inner, (Interval, Zonotope, Array)):
            raise TypeError(f"Unsupported type for inner: {type(inner)}")
        return None

    def contains(self: IntervalType["d"],
                 inner: ContinuousSetType["d"] | Float[Array, "d"],
                 solver: None
                 ) -> Bool[Array, ""]:
        """
        Checks if a continuous set or point is contained in the interval.

        A zonotope is contained exactly when its interval hull is, so both reduce to comparing intervals.

        Args:
            inner: The continuous set or point to check.
            solver: None, as returned by `make_contains`.

        Returns:
            Flag indicating containment, up to `config.tolerance` along each axis.

        Notes:
            Could also override the __contains__ operator.
        """
        from .zonotope import Zonotope

        if isinstance(inner, Zonotope):
            inner = inner.interval()
        if isinstance(inner, Interval):
            return (jnp.abs(inner.centre - self.centre) + inner.radius <= self.radius + config.tolerance).all()
        elif isinstance(inner, Array):
            return (jnp.abs(inner - self.centre) <= self.radius + config.tolerance).all()
        else:
            raise TypeError(f"Unsupported type for inner: {type(inner)}")
