from typing import Annotated, Any, Protocol, runtime_checkable, Callable, ClassVar

from beartype.vale import Is
from jaxtyping import Float, Array, Bool, PRNGKeyArray, PyTree

from .solver import Solver


class ContinuousSetTypeClass:
    r"""
    Subscriptable annotation factory for continuous sets, mirroring `Float[Array, "d"]`:
    `ContinuousSetTypeClass(...)["d"]` annotates a set of dimension `d`.

    A continuous set is a pytree of arrays. Every leaf shares the leading dimension `d`
    while its remaining axes (e.g. the number of generators) are left free, so
    `PyTree[Float[Array, "d ..."], structure]` captures exactly "a set of dimension d".
    The dimension name is bound per call in the surrounding `jaxtyped` context, hence
    `X["d"]` on `self` and `X["m"]` on the return relates the two dimensions.

    Args:
        structure: Pytree structure name; leave `None` for "any continuous set". With a
                   name, every annotation sharing it within one call must have the same
                   pytree structure. Since the structure includes the registered node
                   type, anchoring on `self` pins the arguments to the same class.

    Notes:
        Must be defined before the class it annotates: annotations are evaluated when the
        method is decorated, i.e. while the class body is still executing, and a string
        forward reference silently degrades to `Any` at that point.
    """

    def __init__(self, structure: str | None = None):
        self._structure = structure

    def __getitem__(self, dim: str) -> Any:
        leaf = Float[Array, f"{dim} ..."]
        shape = PyTree[leaf] if self._structure is None else PyTree[leaf, self._structure]
        return Annotated[shape, Is[lambda x: isinstance(x, ContinuousSet)]]  # the protocol is defined below


ContinuousSetType = ContinuousSetTypeClass()
ZonotopeType = ContinuousSetTypeClass("Zonotope")
IntervalType = ContinuousSetTypeClass("Interval")
# TODO: a ContinuousSetTypeClass("Polytope") once Polytope is a continuous set: it lacks most of the protocol, and
#  its anchors have no leading dimension d.
PolytopeType = PyTree[Float[Array, "..."], "Polytope"]


@runtime_checkable
class ContinuousSet(Protocol):
    random: ClassVar[Callable[..., "ContinuousSet"]]

    def sample(self: ContinuousSetType["d"],
               key: PRNGKeyArray,
               num_samples: int,
               ) -> Float[Array, "{num_samples} d"]:
        r"""
        Sample from the continuous set.

        Args:
            key: PRNG key.
            num_samples: The number of samples to draw.

        Returns:
            Sampled points.
        """
        ...

    def support(self: ContinuousSetType["d"],
                direction: Float[Array, "d"]
                ) -> Float[Array, ""]:
        r"""
        Compute the support of the continuous set in the given direction.

        Args:
            direction: The direction in which to compute the support, expected to be of unit length.

        Returns:
            Support in the given direction.
        """
        ...

    def make_contains(self: ContinuousSetType["d"],
                      inner: ContinuousSetType["d"] | Float[Array, "d"],
                      ) -> Solver | None:
        r"""
        Instantiate the solver for the containment LP, reusable across calls with the same shapes and type.

        Args:
            inner: An example of the kind of continuous set or point, whose containment to check.

        Returns:
            Solver instance, or None if the check needs none; consume it through `contains`.
        """
        ...

    def contains(self: ContinuousSetType["d"],
                 inner: ContinuousSetType["d"] | Float[Array, "d"],
                 solver: Solver | None
                 ) -> Bool[Array, ""]:
        """
        Checks if a continuous set or point is contained in the continuous set.

        The solver must have been constructed via `make_contains` with an `inner` of the same type
        and shape, and with a continuous set of the same shape as `self`.

        Args:
            inner: The continuous set or point to check.
            solver: Solver instance.

        Returns:
            Flag indicating containment.

        Notes:
            Could also override the __contains__ operator.
        """
        ...
