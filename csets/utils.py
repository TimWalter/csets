import dataclasses

import jax
import jax.numpy as jnp

from jaxtyping import Array, Float


def pytree_dataclass(cls: type) -> type:
    """
    Annotator that makes `cls` a frozen, keyword-only dataclass whose fields are the children of a JAX pytree.

    Args:
        cls: The class to turn into a pytree dataclass.

    Returns:
        The same class as a registered pytree dataclass.
    """
    cls = dataclasses.dataclass(frozen=True, kw_only=True)(cls)
    names = tuple(field.name for field in dataclasses.fields(cls))

    def unflatten(_, children):
        obj = object.__new__(cls)
        for name, child in zip(names, children):
            object.__setattr__(obj, name, child)
        return obj

    jax.tree_util.register_pytree_with_keys(
        cls,
        lambda obj: (tuple((jax.tree_util.GetAttrKey(name), getattr(obj, name)) for name in names), None),
        unflatten,
        flatten_func=lambda obj: (tuple(getattr(obj, name) for name in names), None))
    return cls


def safe_norm(x: Float[Array, "..."], axis: int, keepdims: bool = False, fill: float = 1.0) -> Float[Array, "..."]:
    """
    Compute the Euclidean norm of a matrix or vector with a NaN-free backward pass on zero vectors.

    `jnp.linalg.norm` NaNs the backward pass at exactly zero (the infinite sqrt gradient meets the
    zero cotangent), and masking the *output* with `where` does not help because gradients flow
    through the discarded branch. The primal is sanitised before the sqrt instead, so zero vectors
    take a constant branch whose gradient is exactly zero. Away from zero the gradient of the norm
    is the unit vector, which stays bounded, so no near-zero threshold is needed.

    Args:
        x: N-dimensional array for which the norm will be computed.
        axis: integer or sequence of integers specifying the axes over which the norm
          will be computed. For a single axis, compute a vector norm. For two axes,
          compute a matrix norm. Defaults to all axes of ``x``.
        keepdims: if True, the output array will have the same number of dimensions as
          the input, with the size of reduced axes replaced by ``1`` (default: False).
        fill: Norm reported for zero vectors (e.g. 1.0 when the result is used as a divisor).

    Returns:
        Norms, with zero-vector norms replaced by `fill`.
    """
    squared = jnp.sum(x ** 2, axis=axis, keepdims=keepdims)
    is_zero = squared == 0.0
    norm = jnp.sqrt(jnp.where(is_zero, 1.0, squared))
    return jnp.where(is_zero, fill, norm)


def generalised_cross(vectors: Float[Array, "d d-1"]) -> Float[Array, "d"]:
    r"""
    Compute the generalised cross-product of $d-1$ vectors in $d$ dimensions.

    It is the vector $h$ with $h^\top x = \det[x \mid v_1 \cdots v_{d-1}]$ for every $x$, so
    $h_i = \det[e_i \mid v_1 \cdots v_{d-1}]$. It is orthogonal to every $v_j$, since the determinant then has a
    repeated column, its length is the $(d-1)$-volume of the parallelotope the vectors span, and it is zero
    exactly when they are linearly dependent. In three dimensions it is the cross-product, `jnp.cross`.

    Args:
        vectors: The $d-1$ vectors, as columns.

    Returns:
        The generalised cross-product.
    """
    d = vectors.shape[0]
    return jax.vmap(lambda e: jnp.linalg.det(jnp.column_stack([e, vectors])))(jnp.eye(d))
