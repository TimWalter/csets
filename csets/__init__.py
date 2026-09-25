from jaxtyping import install_import_hook

# Applies `@jaxtyped(typechecker=beartype)` to every function and dataclass in the submodules.
# The hook rewrites modules as they are imported, so the imports must happen inside the block.
with install_import_hook("csets", "beartype.beartype"):
    from .settings import Config, config
    from .solver import Solver
    from .types import ContinuousSetTypeClass, ContinuousSetType, ContinuousSet
    from .polytope import Polytope
    from .interval import Interval
    from .zonotope import Zonotope

__all__ = ["Config", "config", "Solver", "ContinuousSetTypeClass", "ContinuousSetType", "ContinuousSet", "Interval", "Polytope", "Zonotope"]
