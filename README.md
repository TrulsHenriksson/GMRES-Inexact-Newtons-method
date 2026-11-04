# GMRES + Inexact Newton's method
Two numerical algorithms, together capable of finding roots to very large systems of equations efficiently, i.e. finding $\mathbf{x} \in \mathbb{R}^N$ such that $\mathbf{f}(\mathbf{x}) = \mathbf{0}$, where $N$ can be very large.

The Inexact Newton method linearizes the function $\mathbf{f}$, and solves the linearization approximately (hence "inexact") using GMRES. Since GMRES doesn't need the Jacobian, it is ideal as a sub-solver for very large systems. This is iterated, with the GMRES sub-solver tolerance set by the Eisenstat-Walker approach, until the specified tolerance is met.

These algorithms were written as part of the course NUMN28 Numerical Simulation of Flow Problems at Lund University.

The functions in `debug_table.py` can be useful as part of many different numerical algorithms, and can be found on their own at [this GitHub Gist](https://gist.github.com/TrulsHenriksson/dc7a5b3dfc40693095255cfbcef9fde6).

## Goal
The main goal of this project was referentiability. I wanted to understand these algorithms really well myself, and so it was really important to me that I should make them well-documented and readable. I hope that anyone interested in understanding them will find them simpler to understand after studying this code.