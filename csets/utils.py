import jax.numpy as jnp

from jaxtyping import Array, Float


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
