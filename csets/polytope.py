from jaxtyping import Array, Float, Bool

from .settings import config
from .utils import pytree_dataclass


@pytree_dataclass
class Polytope:
    r"""
    Polytope in halfspace representation, $\{x \mid a_i^\top x \leq b_i\}$.

    Attributes:
        normal: The halfspaces' normals $a_i$.
        anchor: The halfspaces' offsets $b_i$ along their normals.
    """
    normal: Float[Array, "n d"]
    anchor: Float[Array, "n"]

    def contains(self, point: Float[Array, "d"]) -> Bool[Array, ""]:
        """
        Check whether a point is contained in the polytope.

        Args:
            point: The point to check.

        Returns:
            Flag indicating containment.
        """
        return (self.normal @ point <= self.anchor + config.tolerance).all()
