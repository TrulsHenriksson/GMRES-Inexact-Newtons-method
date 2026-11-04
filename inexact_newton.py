import numpy as np

from typing import Callable

from debug_table import debug_table_header, debug_table_row
from vector_functions import Vector, two_norm, linearize
from gmres import gmres



@np.errstate(over="raise", divide="raise", invalid="raise")
def inexact_newton(
    function: Callable[[Vector], Vector],
    x0: Vector,
    *,
    linearization_at: Callable[[Vector], Callable[[Vector], Vector]] | None = None,
    precond_at: Callable[[Vector], Callable[[Vector], Vector]] | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    max_iterations: int = 100,
    raise_if_exceeded: bool = True,
    gamma: float = 0.5,
    eta_max: float = 0.999,
    debug_level: int = 0,
) -> tuple[Vector, int, int]:
    """Find a root of `function` close to `x0` using the inexact Newton method.

    Uses GMRES as a subsolver, which makes this method ideal for large systems.

    Arguments
    ---------
    function : (Vector) -> Vector
        Function to find the root of, i.e. finding `x` such that
        `np.allclose(function(x), 0.0) == True`.
    x0 : Vector
        Initial guess for `x`.
    linearization_at : (x: Vector) -> ((Vector) -> Vector) or None
        Function that returns a linearization of `function` around `x`,
        i.e. a linear map that emulates a matrix-vector product with
        the Jacobian of `function` at `x`, and which is used in the GMRES
        sub-solver. If not given, this is instead approximated numerically
        which may affect convergence negatively.
    precond_at : (x: Vector) -> ((v_guess: Vector) -> v_better: Vector) or None
        Function that returns a right preconditioner at `x` approximating
        `J(x) @ v == b` for the GMRES subsolver.
    rtol, atol : float
        Relative and absolute tolerances. Guarantees that
        `two_norm(function(x)) <= rtol * two_norm(function(x0)) + atol`.
    max_iterations : int
        Maximum Newton iterations.
    raise_if_exceeded : bool
        Whether to raise a `RuntimeError` if the maximum iteration number
        is exceeded.
    gamma, eta_max : float
        Parameters for the Eisenstat-Walker control of the GMRES rtol.
    debug_level : int
        Detail level of debug output. At `>= 1` info is printed each iteration.
        The value `debug_level - 1` is passed to the GMRES sub-solver.

    Returns
    -------
    x : Vector
        Vector such that `two_norm(function(x)) < rtol * two_norm(function(x0)) + atol`.
    newton_iterations : int
        Number of iterations that were performed (at least 1).
    total_gmres_iterations : int
        Number of GMRES iterations that were performed in total (at least 1).
    """
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if not 0 < eta_max < 1:
        raise ValueError("eta_max must be in (0, 1)")

    # x will be edited in-place, and we shouldn't edit x0
    x = x0.copy()
    fx = function(x)
    residual_norm = two_norm(fx)
    convergence_tolerance = rtol * residual_norm + atol

    # Initial rtol for the GMRES sub-solver
    eta = eta_max

    # Print debug table
    if debug_level >= 1:
        debug_table_header(("Newton iteration", "^18"), ("Residual norm", "^14"), ("GMRES rtol", "^14"), ("Restarts", "^8"), ("Iterations", "^10"))
        debug_table_row((0, ">18"), (residual_norm, "<14.8g"), (eta, "<14.8g"), ("", ">8"), ("", ">10"))
    # If debug_level >= 2, GMRES will print its own debug table, which should be indented for clarity
    gmres_debug_indent = " ┃ ".join(" " * width for width in [18, 14, 14, 8, 10]) if debug_level >= 2 else ""

    i = 0  # So it can't be unbound
    total_gmres_iterations = 0
    for i in range(max_iterations):
        # Linearize numerically if exact linearizer is not given
        linearization = linearization_at(x) if linearization_at is not None else linearize(function, x, fx)
        # The preconditioner knows where the linearization originates at
        r_precond = precond_at(x) if precond_at is not None else None

        ### Newton step: solve J(x)s = -f(x) for s, where J(x) is the linearization (Jacobian) of f at x.
        step, gmres_restarts, gmres_iterations = gmres(
            linearization,
            -fx,
            r_precond=r_precond,
            rtol=eta,
            raise_if_exceeded=raise_if_exceeded,
            debug_level=debug_level - 1,
            debug_indent=gmres_debug_indent,
        )
        total_gmres_iterations += gmres_iterations

        x += step
        fx = function(x)
        old_residual_norm = residual_norm
        residual_norm = two_norm(fx)

        if residual_norm < convergence_tolerance:
            # Converged
            if debug_level >= 1:
                debug_table_row((i + 1, ">18"), (residual_norm, "<14.8g"), ("", "<14"), (gmres_restarts, ">8"), (gmres_iterations, ">10"))
            break

        ### Update eta, the rtol for the GMRES sub-solver
        # Bound eta below if it was relatively large (so it can't decrease too suddenly)
        eta_min = gamma * eta**2
        if eta_min <= 0.1:
            eta_min = 0.0

        eta = gamma * (residual_norm / old_residual_norm) ** 2  # New value
        eta = min(eta, eta_max)  # Bound above by the fixed upper bound
        eta = max(eta, eta_min)  # Bound below so as not to decrease too suddenly
        eta = max(eta, 0.5 * convergence_tolerance / residual_norm)  # Bound below to not over-solve in the final stages

        if debug_level >= 1:
            debug_table_row((i + 1, ">18"), (residual_norm, "<14.8g"), (eta, "<14.8g"), (gmres_restarts, ">8"), (gmres_iterations, ">10"))
    else:
        if raise_if_exceeded:
            raise RuntimeError(f"Newton's method did not converge in {max_iterations} iterations.")

    return x, i + 1, total_gmres_iterations