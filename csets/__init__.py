from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Callable, Any, ClassVar

from jaxtyping import Float, Array, Bool, PRNGKeyArray, PyTree, install_import_hook
from moreau.jax import Solver, Settings


@dataclass
class Config:
    enable_grad: bool = True
    device: str | None = None  # Moreau device ('cpu' or 'cuda'); None picks by problem size.


config = Config()


def _jax_has_cuda() -> bool:
    import jax
    try:
        return bool(jax.devices("cuda"))
    except RuntimeError:
        return False


class MoreauSolver(Solver):
    """Moreau solver with a customised auto-tune procedure, such that it is executed at construction time."""

    def __init__(self,
                 n: int,
                 auto_tune_call: Callable[[Solver], Any] | None = None,
                 settings: Settings | None = None,
                 enable_grad: bool | None = None,
                 **kwargs):
        """
        Requires keyword arguments.

        Args:
            n: Number of primal variables
            auto_tune_call: First solve for auto-tuning.
            settings: Optional solver settings (moreau.Settings object).
            enable_grad: Whether the solve is differentiable; defaults to `config.enable_grad`.
                         False where only a decision is read off the solution.
        """
        if settings is None:
            settings = Settings()
        if config.device is not None:
            settings.device = config.device
        else:
            # Moreau to eagerly assign GPU, but only one JAX can use: Moreau's CUDA path needs JAX's.
            settings.device = 'cpu' if n < 500 or not _jax_has_cuda() else 'auto'
        settings.enable_grad = config.enable_grad if enable_grad is None else enable_grad
        super().__init__(n=n, settings=settings, **kwargs)
        if auto_tune_call is not None:
            auto_tune_call(self)


class SetType:
    r"""
    Subscriptable annotation factory for continuous sets, mirroring `Float[Array, "d"]`:
    `SetType(...)["d"]` annotates a set of dimension `d` and nothing else.

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

    def __getitem__(self, dim: str) -> type:
        leaf = Float[Array, f"{dim} ..."]
        return PyTree[leaf] if self._structure is None else PyTree[leaf, self._structure]


ContinuousSetType = SetType()


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
                      inner: ContinuousSetType["d"] | Float[Array, "*n d"],
                      ) -> Any:
        r"""
        Set up the containment check, reusable across calls as long as the shapes and type stay the same.

        Args:
            inner: An example of the kind of continuous set or point(s), whose containment to check.

        Returns:
            What `contains` consumes: a Moreau solver, or for points a `PointContainment`.
        """
        ...

    def contains(self: ContinuousSetType["d"],
                 inner: ContinuousSetType["d"] | Float[Array, "*n d"],
                 solver: Any
                 ) -> Bool[Array, "*n"]:
        """
        Checks if a continuous set, a point, or each of several points is contained in the continuous set.

        The solver must have been constructed via `make_contains` with an `inner` of the same type
        (and, for sets, shape), and with a continuous set of the same shape as `self`.

        Args:
            inner: The continuous set, point (d,) or points (n, d) to check.
            solver: What `make_contains` returned.

        Returns:
            Flag indicating containment, one per point for several points.

        Notes:
            Could also override the __contains__ operator.
        """
        ...


# Applies `@jaxtyped(typechecker=beartype)` to every function and dataclass in the
# submodules below, so the annotations do not have to be decorated one by one. The hook
# rewrites modules as they are imported, so the imports must happen inside the block.
with install_import_hook("csets", "beartype.beartype"):
    from .zonotope import Zonotope
