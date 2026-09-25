import math
import functools

from typing import Callable

import jax
import cvxpy as cp
import jax.numpy as jnp

from jaxtyping import Array

Node = cp.Expression | cp.constraints.constraint.Constraint | cp.problems.objective.Objective
Function = Callable[[tuple[Array, ...]], Array]


def to_jax(node: Node, leaves: list[cp.Variable | cp.Parameter]) -> Function:
    """
    Compile a CVXPY node into a jax.numpy function of the values of the leaves.

    Args:
        node: An expression, an objective, or a constraint.
        leaves: The variables and parameters whose values the function takes.

    Returns:
        A function from the leaves' values to the node's value.
    """
    if isinstance(node, (cp.Variable, cp.Parameter)):
        position = [leaf.id for leaf in leaves].index(node.id)  # ids: == on CVXPY leaves builds a constraint
        return lambda values: values[position]
    if isinstance(node, cp.Constant):
        value = node.value
        value = value.toarray() if hasattr(value, "toarray") else value
        return lambda values: jnp.asarray(value)
    name = type(node).__name__
    if name not in CVXPY_TO_JAX:
        raise NotImplementedError(f"{name} has no jax.numpy translation; see csets/solver/compiler.py")
    translation = CVXPY_TO_JAX[name]
    arguments = [to_jax(argument, leaves) for argument in node.args]
    return lambda values: translation(node, [argument(values) for argument in arguments])


CVXPY_TO_JAX = {
    # arithmetic
    "AddExpression": lambda atom, values: functools.reduce(jnp.add, values),
    "NegExpression": lambda atom, values: -values[0],
    "MulExpression": lambda atom, values: (values[0] * values[1] if jnp.ndim(values[0]) == 0 or jnp.ndim(values[1]) == 0
                                           else values[0] @ values[1]),
    "DivExpression": lambda atom, values: values[0] / values[1],
    "multiply": lambda atom, values: values[0] * values[1],
    # structure
    "index": lambda atom, values: values[0][atom._orig_key],
    "special_index": lambda atom, values: values[0][atom.key],
    "Promote": lambda atom, values: jnp.broadcast_to(values[0], atom.shape),
    "broadcast_to": lambda atom, values: jnp.broadcast_to(values[0], atom.broadcast_shape),
    "reshape": lambda atom, values: jnp.reshape(values[0], atom.shape, order=atom.order),
    "transpose": lambda atom, values: jnp.transpose(values[0], atom.axes),
    "Hstack": lambda atom, values: jnp.hstack(values),
    "Vstack": lambda atom, values: jnp.vstack(values),
    "Concatenate": lambda atom, values: jnp.concatenate(values, axis=atom.axis),
    "diag_vec": lambda atom, values: jnp.diag(values[0], k=atom.k),
    "diag_mat": lambda atom, values: jnp.diagonal(values[0], offset=atom.k),
    "upper_tri": lambda atom, values: jnp.reshape(values[0][jnp.triu_indices(values[0].shape[-1], k=1)], atom.shape),
    "Trace": lambda atom, values: jnp.trace(values[0]),
    "cumsum": lambda atom, values: jnp.cumsum(values[0], axis=atom.axis),
    "cumprod": lambda atom, values: jnp.cumprod(values[0], axis=atom.axis),
    "cummax": lambda atom, values: jax.lax.cummax(values[0], axis=atom.axis % values[0].ndim),
    "kron": lambda atom, values: jnp.kron(values[0], values[1]),
    "conv": lambda atom, values: _conv(values),
    "convolve": lambda atom, values: jnp.convolve(values[0], values[1]),
    # wraps: the value unchanged, only CVXPY's view of it (PSD, symmetric, ...) differs
    **{wrap: lambda atom, values: values[0] for wrap in ("Wrap", "psd_wrap", "nsd_wrap", "symmetric_wrap",
                                                         "hermitian_wrap", "skew_symmetric_wrap", "nonneg_wrap",
                                                         "nonpos_wrap")},
    # elementwise
    "abs": lambda atom, values: jnp.abs(values[0]),
    "sign": lambda atom, values: jnp.where(values[0] > 0, 1.0, -1.0),
    "ceil": lambda atom, values: jnp.ceil(jnp.round(values[0], _EVAL_DECIMALS)),
    "floor": lambda atom, values: jnp.floor(jnp.round(values[0], _EVAL_DECIMALS)),
    "maximum": lambda atom, values: functools.reduce(jnp.maximum, values),
    "minimum": lambda atom, values: functools.reduce(jnp.minimum, values),
    "one_minus_pos": lambda atom, values: 1 - values[0],
    "exp": lambda atom, values: jnp.exp(values[0]),
    "xexp": lambda atom, values: values[0] * jnp.exp(values[0]),
    "log": lambda atom, values: jnp.log(values[0]),
    "log1p": lambda atom, values: jnp.log1p(values[0]),
    "logistic": lambda atom, values: jnp.logaddexp(0, values[0]),
    "entr": lambda atom, values: jax.scipy.special.entr(values[0]),
    "rel_entr": lambda atom, values: jax.scipy.special.rel_entr(values[0], values[1]),
    "kl_div": lambda atom, values: jax.scipy.special.kl_div(values[0], values[1]),
    "huber": lambda atom, values: _huber(getattr(atom.M, "value", atom.M), values[0]),
    "normcdf": lambda atom, values: jax.scipy.stats.norm.cdf(values[0]),
    "PowerApprox": lambda atom, values: values[0] ** getattr(atom.p, "value", atom.p),
    "Power": lambda atom, values: values[0] ** getattr(atom.p, "value", atom.p),
    "sin": lambda atom, values: jnp.sin(values[0]),
    "cos": lambda atom, values: jnp.cos(values[0]),
    "tan": lambda atom, values: jnp.tan(values[0]),
    "sinh": lambda atom, values: jnp.sinh(values[0]),
    "tanh": lambda atom, values: jnp.tanh(values[0]),
    "asinh": lambda atom, values: jnp.arcsinh(values[0]),
    "atanh": lambda atom, values: jnp.arctanh(values[0]),
    "real": lambda atom, values: jnp.real(values[0]),
    "imag": lambda atom, values: jnp.imag(values[0]),
    "conj": lambda atom, values: jnp.conj(values[0]),
    # logic, on 0/1 values
    "Not": lambda atom, values: 1 - values[0],
    "And": lambda atom, values: functools.reduce(jnp.minimum, values),
    "Or": lambda atom, values: functools.reduce(jnp.maximum, values),
    "Xor": lambda atom, values: functools.reduce(lambda a, b: jnp.mod(a + b, 2), values),
    # reductions
    "Sum": lambda atom, values: jnp.sum(values[0], axis=atom.axis, keepdims=atom.keepdims),
    "max": lambda atom, values: jnp.max(values[0], axis=atom.axis, keepdims=atom.keepdims),
    "min": lambda atom, values: jnp.min(values[0], axis=atom.axis, keepdims=atom.keepdims),
    "Prod": lambda atom, values: jnp.prod(values[0], axis=atom.axis, keepdims=atom.keepdims),
    "sum_largest": lambda atom, values: _sum_largest(atom, values[0]),
    "dotsort": lambda atom, values: _dotsort(values[0], values[1]),
    "length": lambda atom, values: jnp.max(
        jnp.where(jnp.abs(values[0]) > cp.settings.ATOM_EVAL_TOL, jnp.arange(1, values[0].size + 1), 0)),
    "norm1": lambda atom, values: jnp.sum(jnp.abs(values[0]), axis=atom.axis, keepdims=atom.keepdims),
    "norm_inf": lambda atom, values: jnp.max(jnp.abs(values[0]), axis=atom.axis, keepdims=atom.keepdims),
    "PnormApprox": lambda atom, values: _pnorm(atom, values[0]),
    "Pnorm": lambda atom, values: _pnorm(atom, values[0]),
    "GeoMeanApprox": lambda atom, values: _geo_mean(atom, values[0]),
    "GeoMean": lambda atom, values: _geo_mean(atom, values[0]),
    "dist_ratio": lambda atom, values: jnp.linalg.norm(values[0] - atom.a) / jnp.linalg.norm(values[0] - atom.b),
    "quad_over_lin": lambda atom, values: jnp.sum(jnp.abs(values[0]) ** 2, axis=atom.axis, keepdims=atom.keepdims) / values[1],
    "QuadForm": lambda atom, values: jnp.real(jnp.dot(jnp.conj(values[0]).T, values[1] @ values[0])),
    "MatrixFrac": lambda atom, values: _matrix_frac(*values),
    "log_sum_exp": lambda atom, values: jax.scipy.special.logsumexp(values[0], axis=atom.axis, keepdims=atom.keepdims),
    "gmatmul": lambda atom, values: jnp.exp(jnp.asarray(atom.A.value) @ jnp.log(values[0])),
    "perspective": lambda atom, values: _perspective(atom, values),
    # matrix functions
    "eye_minus_inv": lambda atom, values: jnp.linalg.inv(jnp.eye(values[0].shape[0]) - values[0]),
    "sigma_max": lambda atom, values: jnp.linalg.norm(values[0], 2),
    "normNuc": lambda atom, values: jnp.linalg.norm(values[0], "nuc"),
    "lambda_max": lambda atom, values: jnp.linalg.eigvalsh(values[0]).max(),
    "lambda_sum_largest": lambda atom, values: _largest_sum(atom.k, jnp.linalg.eigvalsh(values[0])),
    "gen_lambda_max": lambda atom, values: _gen_lambda_max(*values),
    "condition_number": lambda atom, values: (lambda eigenvalues: eigenvalues[-1] / eigenvalues[0])(
        jnp.linalg.eigvalsh(values[0])),
    "pf_eigenvalue": lambda atom, values: jnp.abs(jnp.linalg.eigvals(values[0])).max(),
    "log_det": lambda atom, values: _log_det(values[0]),
    "tr_inv": lambda atom, values: _tr_inv(values[0]),
    "von_neumann_entr": lambda atom, values: jnp.sum(jax.scipy.special.entr(jnp.linalg.eigvalsh(values[0]))),
    "quantum_rel_entr": lambda atom, values: _quantum_rel_entr(atom, *values),
    # objectives: their expression's value
    "Minimize": lambda objective, values: values[0],
    "Maximize": lambda objective, values: values[0],
    # constraints: the largest residual
    "Equality": lambda constraint, values: jnp.max(jnp.abs(values[0] - values[1])),
    "Zero": lambda constraint, values: jnp.max(jnp.abs(values[0])),
    "Inequality": lambda constraint, values: jnp.max(jnp.maximum(values[0] - values[1], 0)),
    "NonPos": lambda constraint, values: jnp.max(jnp.maximum(values[0], 0)),
    "NonNeg": lambda constraint, values: jnp.max(jnp.maximum(-values[0], 0)),
    "PSD": lambda constraint, values: jnp.max(jnp.maximum(
        -jnp.linalg.eigvalsh((values[0] + jnp.swapaxes(values[0], -2, -1)) / 2).min(axis=-1), 0)),
    "FiniteSet": lambda constraint, values: jnp.max(jnp.min(
        jnp.abs(values[0].reshape(-1, 1, order="F") - values[1].reshape(1, -1)), axis=1)),
    "SOC": lambda constraint, values: jnp.max(_soc_residual(constraint, values)),
    "ExpCone": lambda constraint, values: jnp.max(_exp_cone_residual(values)),
    "PowCone3D": lambda constraint, values: jnp.max(_pow_cone_3d_residual(constraint, values)),
    "PowCone3DApprox": lambda constraint, values: jnp.max(_pow_cone_3d_residual(constraint, values)),
    "PowConeND": lambda constraint, values: jnp.max(_pow_cone_nd_residual(constraint, values)),
    "RelEntrConeQuad": lambda constraint, values: jnp.max(_rel_entr_cone_residual(values)),
}

# ceil and floor first round to the decimals of CVXPY's evaluation tolerance.
_EVAL_DECIMALS = int(abs(math.log10(cp.settings.ATOM_EVAL_TOL)))


def _pnorm(atom, x):
    """
    Compute the p-norm (sum |x|^p)^(1/p).

    Args:
        atom: The Pnorm or PnormApprox atom, giving p, the axis and keepdims.
        x: The argument's value.

    Returns:
        The p-norm over the atom's axis, or over all entries if it is None.
    """
    return jnp.sum(jnp.abs(x) ** atom.p, axis=atom.axis, keepdims=atom.keepdims) ** (1 / atom.p)


def _conv(values):
    """
    Convolve two vectors.

    Args:
        values: The values of the two arguments, vectors or column vectors.

    Returns:
        The convolution of the flattened arguments, as a column if either argument is a matrix.
    """
    output = jnp.convolve(values[0].ravel(), values[1].ravel())
    return output[:, None] if values[0].ndim == 2 or values[1].ndim == 2 else output


def _huber(M, x):
    """
    Compute the Huber function elementwise.

    Args:
        M: The threshold between the quadratic and the linear part.
        x: The argument's value.

    Returns:
        x^2 where |x| <= M, 2 M |x| - M^2 elsewhere.
    """
    return jnp.where(jnp.abs(x) <= M, x ** 2, 2 * M * jnp.abs(x) - M ** 2)


def _largest_sum(k, x):
    """
    Sum the k largest entries along the last axis, for a possibly fractional k.

    Args:
        k: How many entries to sum; its fractional part weights the next largest entry.
        x: The values, reduced along their last axis.

    Returns:
        The sum of the floor(k) largest entries plus the fractional part of k times the next largest one.
    """
    k_floor = int(k)
    descending = -jnp.sort(-x, axis=-1)
    total = descending[..., :k_floor].sum(axis=-1)
    if k - k_floor > 0 and k_floor < x.shape[-1]:
        total = total + (k - k_floor) * descending[..., k_floor]
    return total


def _sum_largest(atom, x):
    """
    Sum the k largest entries.

    Args:
        atom: The sum_largest atom, giving k, the axis, and keepdims.
        x: The argument's value.

    Returns:
        The sum of the k largest entries over the atom's axis, or over all entries if it is None.
    """
    axes = tuple(range(x.ndim)) if atom.axis is None else (atom.axis,) if isinstance(atom.axis, int) else tuple(atom.axis)
    kept = [a for a in range(x.ndim) if a not in axes]
    moved = jnp.transpose(x, kept + list(axes))
    total = _largest_sum(atom.k, moved.reshape(moved.shape[:len(kept)] + (-1,)))
    return jnp.reshape(total, atom.shape)


def _dotsort(X, W):
    """
    Compute the inner product of the sorted entries of X and W.

    Args:
        X: The value of the first argument.
        W: The value of the second argument, with at most as many entries as X.

    Returns:
        The inner product of X's sorted entries with W's, padded with zeros to X's size and sorted.
    """
    x = X.ravel()
    w = jnp.zeros_like(x).at[:W.size].set(W.ravel())
    return jnp.sort(x) @ jnp.sort(w)


def _geo_mean(atom, x):
    """
    Compute the weighted geometric mean.

    Args:
        atom: The GeoMean or GeoMeanApprox atom, giving the weights, which entries they apply to, the axis, and
         keepdims.
        x: The argument's value.

    Returns:
        The weighted geometric mean over the atom's axis, or over all entries if it is None; entries with weight
        zero take no part.
    """
    w = jnp.asarray([float(w) for w in atom.w])                  # fractions
    if atom.axis is None:
        out = jnp.prod(x.ravel(order="F")[atom._keep] ** w)
    else:
        axes = (atom.axis,) if isinstance(atom.axis, int) else tuple(atom.axis)
        moved = jnp.moveaxis(x, axes, list(range(len(axes))))
        flat = moved.reshape((atom._reduced_size(), -1), order="F")
        out = jnp.prod(flat[atom._keep] ** w[:, None], axis=0).reshape(moved.shape[len(axes):], order="F")
    return jnp.reshape(out, atom.shape, order="F")


def _matrix_frac(X, P):
    """
    Compute tr(X^H P^-1 X).

    Args:
        X: The value of the first argument, a vector or a matrix.
        P: The value of the second argument, a positive definite matrix.

    Returns:
        tr(X^H P^-1 X), or x^H P^-1 x for a vector.
    """
    product = jnp.conj(X).T @ jnp.linalg.solve(P, X)
    return jnp.trace(product) if product.ndim == 2 else product


def _perspective(atom, values):
    """
    Compute the perspective s f(x / s).

    Args:
        atom: The perspective atom, giving f and, optionally, its recession function f_recession.
        values: The values of the arguments: s, then f's variables.

    Returns:
        s f(x / s), or f_recession(x) at s = 0. CVXPY's own value at s = 0 is 0 whatever f_recession(x) is: it
        evaluates f_recession but multiplies it by s.
    """
    s, xs = values[0], values[1:]
    f = to_jax(atom.f, atom.f.variables())
    if atom.f_recession is None:
        return s * f(tuple(x / s for x in xs))
    f_recession = to_jax(atom.f_recession, atom.f_recession.variables())
    at_zero = jnp.isclose(s, 0)
    s_safe = jnp.where(at_zero, 1, s)
    return jnp.where(at_zero, f_recession(tuple(xs)), s_safe * f(tuple(x / s_safe for x in xs)))


def _gen_lambda_max(A, B):
    """
    Compute the largest generalised eigenvalue.

    Args:
        A: A symmetric matrix.
        B: A positive definite matrix.

    Returns:
        The largest lambda with A x = lambda B x: the largest eigenvalue of L^-1 A L^-T, where B = L L^T.
    """
    L = jnp.linalg.cholesky(B)
    half = jax.scipy.linalg.solve_triangular(L, A, lower=True)
    return jnp.linalg.eigvalsh(jax.scipy.linalg.solve_triangular(L, half.T, lower=True)).max()


def _log_det(A):
    """
    Compute log det.

    Args:
        A: The argument's value.

    Returns:
        The log determinant of A's Hermitian part, or -inf unless that determinant is positive.
    """
    sign, logdet = jnp.linalg.slogdet((A + jnp.conj(A).T) / 2)
    return jnp.where(jnp.isclose(jnp.real(sign), 1), logdet, -jnp.inf)


def _tr_inv(X):
    """
    Compute tr(X^-1).

    Args:
        X: The argument's value.

    Returns:
        The trace of X's inverse, or inf unless X is Hermitian and positive definite.
    """
    eigenvalues = jnp.linalg.eigvalsh((X + X.T) / 2)
    valid = (jnp.linalg.norm(X - jnp.conj(X).T) < 1e-8) & (eigenvalues.min() > 0)
    return jnp.where(valid, jnp.sum(1 / eigenvalues), jnp.inf)


def _quantum_rel_entr(atom, X, Y):
    """
    Compute the quantum relative entropy tr(X (log X - log Y)).

    Args:
        atom: The quantum_rel_entr atom, giving the tolerance on negative eigenvalues.
        X: The value of the first argument.
        Y: The value of the second argument.

    Returns:
        tr(X (log X - log Y)) of the Hermitian parts, or inf unless both are PSD up to the tolerance. CVXPY's own
        value differs unless tr X = 1: it takes tr(X log X) as -scipy.stats.entropy of X's eigenvalues, which
        normalises them to sum to 1 first.
    """
    w1, V = jnp.linalg.eigh((X + jnp.conj(X).T) / 2)
    w2, W = jnp.linalg.eigh((Y + jnp.conj(Y).T) / 2)
    valid = (w1.min() >= -atom.EVAL_TOL) & (w2.min() >= -atom.EVAL_TOL)
    w1, w2 = jnp.maximum(w1, 0), jnp.maximum(w2, 0)
    cross = w1 @ jnp.abs(jnp.conj(V).T @ W) ** 2 @ jnp.log(w2)
    return jnp.where(valid, -jnp.sum(jax.scipy.special.entr(w1)) - cross, jnp.inf)


def _soc_residual(constraint, values):
    """
    Compute the residual of a second-order cone constraint.

    Args:
        constraint: The SOC constraint, giving the axis along which its cones lie.
        values: The values of its arguments: t, and X with one cone per entry of t.

    Returns:
        The distance of each (t, x) to the cone {||x|| <= t}, through the cone's closed-form projection.
    """
    t, X = jnp.atleast_1d(values[0]), jnp.atleast_1d(values[1])
    X = X.T if constraint.axis == 0 else X
    promoted = X.ndim == 1
    X = jnp.atleast_2d(X)                                         # one cone per row
    norms = jnp.linalg.norm(X, axis=1)
    inside, below = t >= norms, t <= -norms
    shrink = 0.5 * (1 + t / jnp.where(norms > 0, norms, 1))       # between the two: onto the cone's surface
    t_projection = jnp.where(inside, t, jnp.where(below, 0, shrink * norms))
    X_projection = jnp.where(inside[:, None], X, jnp.where(below[:, None], 0, shrink[:, None] * X))
    residual = jnp.sqrt(jnp.sum((X - X_projection) ** 2, axis=1) + (t - t_projection) ** 2)
    return residual[0] if promoted else residual


def _exp_cone_residual(values):
    """
    Compute the membership violation of an exponential cone constraint.

    Args:
        values: The values of its arguments x, y and z.

    Returns:
        How far each (x, y, z) violates cl{y > 0, y exp(x / y) <= z}: zero exactly on the cone, positive outside,
        but not the distance, which CVXPY finds by projecting with SCS.
    """
    x, y, z = values
    y_safe = jnp.where(y > 0, y, 1)
    interior = jnp.maximum(y_safe * jnp.exp(x / y_safe) - z, 0)            # y > 0
    closure = jnp.maximum(-y, 0) + jnp.maximum(x, 0) + jnp.maximum(-z, 0)  # y <= 0: only y = 0, x <= 0 <= z
    return jnp.where(y > 0, interior, closure)


def _pow_cone_3d_residual(constraint, values):
    """
    Compute the membership violation of a 3D power cone constraint.

    Args:
        constraint: The PowCone3D or PowCone3DApprox constraint, giving the exponent alpha.
        values: The values of its arguments x, y and z.

    Returns:
        How far each (x, y, z) violates {x^alpha y^(1 - alpha) >= |z|, x >= 0, y >= 0}: zero exactly on the cone,
        positive outside, but not the distance, which CVXPY finds by projecting with SCS.
    """
    x, y, z = values
    alpha = jnp.asarray(constraint.alpha.value)                   # a constant attribute, not an argument
    x_plus, y_plus = jnp.maximum(x, 0), jnp.maximum(y, 0)
    power = x_plus ** alpha * y_plus ** (1 - alpha)
    return jnp.maximum(-x, 0) + jnp.maximum(-y, 0) + jnp.maximum(jnp.abs(z) - power, 0)


def _pow_cone_nd_residual(constraint, values):
    """
    Compute the membership violation of an n-dimensional power cone constraint.

    Args:
        constraint: The PowConeND constraint, giving the exponents alpha and the axis along which its cones lie.
        values: The values of its arguments W and z.

    Returns:
        How far each (w, z) violates {prod(w^alpha) >= |z|, w >= 0}: zero exactly on the cone, positive outside,
        but not the distance, which CVXPY finds by projecting with SCS.
    """
    W, z = values
    alpha = jnp.asarray(constraint.alpha.value)                   # a constant attribute, not an argument
    negative = jnp.sum(jnp.maximum(-W, 0), axis=constraint.axis)
    power = jnp.prod(jnp.maximum(W, 0) ** alpha, axis=constraint.axis)
    return negative + jnp.maximum(jnp.abs(z) - power, 0)


def _rel_entr_cone_residual(values):
    """
    Compute the membership violation of a relative entropy cone constraint.

    Args:
        values: The values of its arguments x, y and z.

    Returns:
        How far each (x, y, z) violates the exact cone cl{x log(x / y) <= z, x > 0, y > 0}, which the quadrature
        constraint approximates: zero exactly on the cone, positive outside, but not the distance, which CVXPY
        finds by projecting with SCS.
    """
    x, y, z = values
    both_positive = (x > 0) & (y > 0)
    x_safe, y_safe = jnp.where(both_positive, x, 1), jnp.where(both_positive, y, 1)
    interior = jnp.maximum(x_safe * jnp.log(x_safe / y_safe) - z, 0)
    closure = jnp.maximum(-x, 0) + jnp.maximum(-y, 0) + jnp.maximum(-z, 0) + jnp.where((x > 0) & (y <= 0), jnp.inf, 0)
    return jnp.where(both_positive, interior, closure)
