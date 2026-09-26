from typing import Callable

import cvxpy as cp
import jax
import jax.numpy as jnp
from cvxpylayers.jax import CvxpyLayer
from jaxtyping import Array, Bool, Float

from .compiler import to_jax

Heuristic = Callable[..., tuple[Bool[Array, ""], tuple[Array, ...]]]


class Solver:
    """
    Solves a parametrised problem as a `CvxpyLayer` and also reports the objective value and the constraints' residuals.
    """

    def __init__(self,
                 problem: cp.Problem,
                 parameters: list[cp.Parameter],
                 variables: list[cp.Variable],
                 heuristic: Heuristic | None = None,
                 **kwargs):
        """
        Args:
            problem: The problem, DPP-compliant in `parameters`.
            parameters: The parameters, in the order the solver is called with.
            variables: The variables to return, in this order. The problem's other variables are solved for too
             but not returned.
            heuristic: A cheaper way to solve a single problem, tried first to avoid proper solves: from the
             parameters' values to whether it succeeded and the values of all of the problem's variables. Where it
             succeeds, its values are returned as they are, so they need not be optimal: only sufficient for
             the heuristic's purpose.
        """
        self._requested = len(variables)
        requested = {x.id for x in variables}  # ids: `in` would compare with ==, which builds a constraint in CVXPY
        all_variables = variables + [x for x in problem.variables() if x.id not in requested]
        if heuristic is not None and len(all_variables) > len(variables):
            raise ValueError("A heuristic must provide all of the problem's variables.")
        self._layer = CvxpyLayer(problem, parameters, all_variables, **kwargs)

        leaves = [*parameters, *all_variables]
        self._objective = to_jax(problem.objective, leaves)
        self._constraints = [to_jax(constraint, leaves) for constraint in problem.constraints]
        self._heuristic = heuristic
        self._solution = self._batched_solution(batch_axes=())

    def __call__(self, *params: Array) -> tuple[tuple[Array, ...], Float[Array, ""], Float[Array, "constraints"]]:
        """
        Args:
            *params: Values of the parameters, in the order given at construction.

        Returns:
            The requested variables' values, and the objective value and constraint residuals at them. The values
            are the layer's solution, optimal up to its accuracy, or the heuristic's where it succeeded.
        """
        solution = self._solution(*params)
        values = (*params, *solution)
        residuals = jnp.stack([constraint(values) for constraint in self._constraints])
        return solution[:self._requested], self._objective(values), residuals

    def _batched_solution(self, batch_axes: tuple[tuple[bool, ...], ...]) -> Callable[..., tuple[Array, ...]]:
        """
        Build the solution of a batch of problems: by the heuristic, and by the layer where it did not solve.

        Args:
            batch_axes: The batch's levels, outermost first: for each, which parameters carry its axis. Parameters
             without it are shared across that level.

        Returns:
            A function of the parameters, batched as `batch_axes`, to the variables' values, with one leading
            axis per level.
        """

        @jax.custom_batching.custom_vmap
        def solution(*params):
            layer_solution = lambda: tuple(_nested(self._layer, batch_axes, sequential=True)(*params))
            if self._heuristic is None:
                return layer_solution()
            solved, heuristic_solution = _nested(self._heuristic, batch_axes, sequential=False)(*params)

            def with_layer():
                return tuple(jnp.where(solved.reshape(solved.shape + (1,) * (h.ndim - solved.ndim)), h, s)
                             for h, s in zip(heuristic_solution, layer_solution()))

            return jax.lax.cond(solved.all(), lambda: tuple(heuristic_solution), with_layer)

        @solution.def_vmap
        def solution_batched(axis_size, in_batched, *params):
            out = self._batched_solution(batch_axes=(tuple(in_batched), *batch_axes))(*params)
            return out, tuple(True for _ in out)

        return solution


def _nested(f: Callable, batch_axes: tuple[tuple[bool, ...], ...], sequential: bool) -> Callable:
    """
    Map a function of a single problem over a batch of problems.

    Args:
        f: The function of a single problem.
        batch_axes: The batch's levels, outermost first: for each, which arguments carry its axis.
        sequential: Whether to map the outer levels one element at a time (`lax.map`) rather than all at once
         (`jax.vmap`); the innermost level is always vectorised. For the layer, which copies every shared
         argument once per problem it solves at once.

    Returns:
        The function of the batch, with one leading output axis per level.
    """
    for depth in reversed(range(len(batch_axes))):
        flags = batch_axes[depth]
        if sequential and depth < len(batch_axes) - 1:
            f = _mapped(f, flags)
        else:
            f = jax.vmap(f, in_axes=[0 if batched else None for batched in flags])
    return f


def _mapped(f: Callable, flags: tuple[bool, ...]) -> Callable:
    """
    Map a function over the leading axis of some of its arguments, one element at a time.

    Args:
        f: The function.
        flags: Which arguments carry the axis; the others are passed unchanged.

    Returns:
        The function mapped with `lax.map`.
    """

    def mapped(*args):
        def one(batched):
            elements = iter(batched)
            return f(*[next(elements) if flag else arg for arg, flag in zip(args, flags)])

        return jax.lax.map(one, [arg for arg, flag in zip(args, flags) if flag])

    return mapped
