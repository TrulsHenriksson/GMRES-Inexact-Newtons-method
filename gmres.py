import numpy as np
from math import sqrt

from typing import Callable

from debug_table import debug_table_header, debug_table_row
from vector_functions import Vector, two_norm



@np.errstate(over="raise", divide="raise", invalid="raise")
def gmres(
    matvec: Callable[[Vector], Vector],
    b: Vector,
    *,
    r_precond: Callable[[Vector], Vector] | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-14,
    max_iterations: int = 20,
    max_restarts: int = 200,
    raise_if_exceeded: bool = True,
    debug_level: int = 0,
    debug_indent: str = "",
) -> tuple[Vector, int, int]:
    r"""Solve the large system `Ax = b` iteratively using the Generalized Minimum RESidual method.

    The problem is solved iteratively by identifying it with the Hessenberg least squares problem

    $\min_{y\in\mathbb{R}^k} \left\lVert g_{\text{RHS}} - \bar{H}_k y \right\rVert$

    where $\bar{H}_k$ is a $(k+1)\times k$ upper Hessenberg matrix. As we go along and populate the matrix, we also
    perform Givens rotations to zero out the sub-diagonal. The resulting upper triangular matrix makes
    it easy to check for convergence and solve for $y$ using backsubstitution.

    Arguments
    ---------
    matvec : (Vector) -> Vector
        Linear function that simulates the matrix-vector product `Ax`. Supplied as a
        function since sparse linear operators are often easily implemented as functions.
    b : Vector
        Right-hand side vector to solve for.
    r_precond : (Vector) -> Vector or None
        Right preconditioner, i.e. function such that `r_precond(b)` is approximately
        equal to the desired `x`.
    rtol, atol : float
        Relative and absolute tolerances. Guarantees that
        `two_norm(matvec(x)) < rtol * two_norm(b) + atol`.
    max_iterations : int
        How many iterations to run for before restarting. Beware that internal storage size
        is on order `max_iterations ** 2`.
    max_restarts : int
        How many times to restart before giving up.
    raise_if_exceeded : bool
        Whether to raise a `RuntimeError` if the maximum number of restarts is exceeded.
    debug_level : int
        Detail level of debug output. At `>= 1` info is printed every restart, and at
        `>= 2` info is printed every iteration.
    debug_indent : str
        String to print before the table. Used by `inexact_newton` to print GMRES tables
        inside the inexact Newton table.

    Returns
    -------
    x : Vector
        Vector such that `two_norm(matvec(x)) < rtol * two_norm(b) + atol`.
    restarts : int
        How many times GMRES had to restart (at least 0).
    total_iterations : int
        How many total iterations were performed (at least 1).
    """
    if r_precond is None:
        r_precond = lambda x: x

    n = b.size
    max_iterations = min(n, max_iterations)

    # Orthonormal basis vectors for the Krylov subspace $\mathcal{K}_k(A, r_0)$
    V = np.empty((max_iterations + 1, *b.shape))
    # Upper Hessenberg (will be turned upper triangular) matrix defining the Hessenberg least squares problem
    H = np.empty((max_iterations + 1, max_iterations))
    # Right-hand side of the Hessenberg least squares problem
    rhs = np.empty(max_iterations + 1)
    # History of Givens rotations that zeroed out H below the diagonal
    givens_rotations = np.empty((max_iterations, 2))

    x = np.zeros(n)  # x0 is the zero vector...
    residual = b.copy()  # ...so $r_0 = b - Ax_0 = b$.
    residual_norm = two_norm(residual)
    convergence_tolerance = rtol * residual_norm + atol

    if debug_level >= 1:
        debug_table_header(("GMRES restarts", "^16"), ("Residual norm", "^14"), ("Iterations", "^16"), indent=debug_indent)
        debug_table_row(("", ">16"), (residual_norm, "<14.8g"), ("", ">16"), indent=debug_indent)

    if residual_norm < convergence_tolerance:
        return x, 0, 0

    converged = False
    total_iterations = 0
    restarts = 0  # So it can't be unbound
    for restarts in range(max_restarts):
        # First vector of the orthonormal basis
        V[0] = residual / residual_norm
        rhs[0] = residual_norm

        j = 0  # So it can't be unbound
        for j in range(max_iterations):  # Iterate over the columns of H
            ### Arnoldi iteration: Gram-Schmidt orthonormalization w.r.t. the set $\{r_0, Ar_0, ..., A^{j-1}r_0\}$.

            Av_j = matvec(r_precond(V[j]))
            H[: j + 1, j] = V[: j + 1] @ Av_j  # $ h_{ij} = \langle Av_j, v_i \rangle $
            w = Av_j - H[: j + 1, j].dot(V[: j + 1])  # $w_{j+1} = Av_j - \sum_{i=1}^j \langle Av_j, v_i \rangle v_i$
            H[j + 1, j] = w_norm = two_norm(w)  # $ h_{j+1,j} = \lVert w_{j+1} \rVert_2 $

            ### The matrix H is now almost upper triangular (except for H[j+1, j]). We
            ### apply one Givens rotation (on H[j, j] and H[j+1, j]) to zero it out.

            # First apply all previous Givens rotations to the new column
            for i, (c, s) in enumerate(givens_rotations[:j]):
                # Equivalent to H[i:i+2, j] = [[c, s], [-s, c]] @ H[i:i+2, j]
                h, h_below = H[i, j], H[i + 1, j]
                H[i, j] =  c * h + s * h_below
                H[i + 1, j] = -s * h + c * h_below

            # Calculate the new Givens rotation
            h, h_below = H[j, j], H[j + 1, j]
            h_norm = sqrt(h**2 + h_below**2)
            c = h / h_norm
            s = h_below / h_norm

            # Apply the new Givens rotation to H. Shortcut: $\begin{bmatrix}c&s\\-s&c\end{bmatrix}\begin{pmatrix}h_{j,j}\\h_{j+1,j}\end{pmatrix} = \begin{pmatrix}\sqrt{h_{j,j}^2+h_{j+1,j}^2}\\0\end{pmatrix}$
            H[j, j] = h_norm
            H[j + 1, j] = 0.0
            givens_rotations[j] = (c, s)

            # The Givens rotation also needs to be applied to the right-hand side vector
            rhs[j + 1] = -s * rhs[j]
            rhs[j] *= c

            if debug_level >= 2:
                debug_table_row(("", ">16"), (abs(rhs[j + 1]), "<14.8g"), (j + 1, ">16"), indent=debug_indent)

            ### We now test for convergence. The newest component of rhs is equal
            ### to the least-squares residual of the Hessenberg problem.
            if abs(rhs[j + 1]) < convergence_tolerance:
                converged = True
                break

            # If we haven't converged yet, we can normalize $v_{j+1}$
            if abs(w_norm) == 0.0:
                raise RuntimeError(f"GMRES got stuck on iteration {j + 1} due to a singular matrix.")
            V[j + 1] = w / w_norm  # $ v_{j+1} = w_{j+1} / \lVert w_{j+1} \rVert_2 $

        total_iterations += j + 1

        ### The solution to the least squares problem is now found by back-substitution,
        ### since H has been turned upper triangular by the Givens rotations.

        # Last component is irrelevant since the last row of H is zero
        for i in reversed(range(j + 1)):
            rhs[i] /= H[i, i]
            rhs[:i] -= H[:i, i] * rhs[i]

        ### The solution to the original problem can now finally be obtained.

        x += V[: j + 1].T @ rhs[: j + 1]  # $ x^{(j)} = x^{(0)} + V_jy $

        if debug_level >= 1:
            debug_table_row((restarts, ">16"), (abs(rhs[j + 1]), "<14.8g"), (j + 1, ">16"), indent=debug_indent)

        if converged:
            break

        residual = b - matvec(r_precond(x))
        residual_norm = two_norm(residual)
    else:
        if raise_if_exceeded:
            raise RuntimeError(f"GMRES did not converge in {max_restarts} restarts.")

    return r_precond(x), restarts, total_iterations
