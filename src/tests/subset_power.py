"""
Some tests for dimensionality reduction where you use random subset of the 
columns to see if power iteration converges faster
"""
import numpy as np
from ..util import power as pwr
from ..util import eig_functs as eig
from ..util import scikit_jl as jl
from ..util.subset import select_d_random_columns

def subset_pow(A, u_0, u_star, num_iter, seed, d):
    """
    A - the matrix (nxm)
    u_0 - an initial guess for the top left eigenvector of A
    u_star - the actual top left eigenvector of A
    num_iter - the number of iterations of power to do
    seed - for repeatable tests
    d - reduction size: A_reduced is (nxd)

    RETURN: xs - a list of iterations [0, 1, 2, ..., num_iter - 1]
            ys - the list of residuals per iteration
    Measure convergence of Power with some random subset of the columns of A
    """

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)

    ys[0] = eig.euclidean_dist(u_0, u_star)
    xs[0] = 0

    A_reduced = select_d_random_columns(A, d, seed)

    v =  pwr.v_from_u(A_reduced, u_0)

    for i in range(1, num_iter):
        # NOTE: using scikit-learn -> top left eig (u) is of significance

        u, _, v = pwr.topsing(v0=v,
                          A=A_reduced, 
                          maxiter=1)
        
        euc_dist = eig.euclidean_dist(u, u_star)
        ys[i] = euc_dist
        xs[i] = i

    return xs, ys, f"column subset {A_reduced.shape}; {} mults"

def percent_subset_pow(A, u_0, u_star, num_iter, seed, p):
    """
    A - the matrix (nxm)
    u_0 - an initial guess for the top left eigenvector of A
    u_star - the actual top left eigenvector of A
    num_iter - the number of iterations of power to do
    seed - for repeatable tests
    p - reduction percentage: A_reduced is (nx(1-p)*m)

    RETURN: xs - a list of iterations [0, 1, 2, ..., num_iter - 1]
            ys - the list of residuals per iteration
    Measure convergence of power with some random subset of the columns of A
    """

    d = jl.percent_reduce(A.shape[1], p)

    xs, ys, label = subset_pow(A, u_0, u_star, num_iter, seed, d)

    return xs, ys, f"{p}% {label}"

def subset_pow_swap(A, u_0, u_star, num_iter, seed, d, step_size):
    """
    A - the matrix (nxm)
    u_0 - an initial guess for the top left eigenvector of A
    u_star - the actual top left eigenvector of A
    num_iter - the number of iterations of Power to do
    seed - for repeatable tests
    d - reduction size: A_reduced is (nxd)
    step_size - how often to regenerate A

    RETURN: xs - a list of iterations [0, 1, 2, ..., num_iter - 1]
            ys - the list of residuals per iteration
    Measure convergence of Power with some random subset of the columns of A
    """

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)

    ys[0] = eig.euclidean_dist(u_0, u_star)
    xs[0] = 0

    A_reduced = select_d_random_columns(A, d, seed)

    v =  pwr.v_from_u(A_reduced, u_0)

    for i in range(1, num_iter):
        # NOTE: using scikit-learn -> top left eig (u) is of significance
        if (i % step_size == 0):
            # Randomly regenerate A
            reduced_A = select_d_random_columns(A, d=d, seed=seed*i)
            v = pwr.v_from_u(reduced_A, u)

        u, _, v = pwr.topsing(v0=v,
                          A=A_reduced, 
                          maxiter=1)
        
        euc_dist = eig.euclidean_dist(u, u_star)
        ys[i] = euc_dist
        xs[i] = i

    return xs, ys, f"column subset (swapping every {step_size}): {A_reduced.shape}"

def percent_subset_pow_swap(A, u_0, u_star, num_iter, seed, p, step_size):
    """
    A - the matrix (nxm)
    u_0 - an initial guess for the top left eigenvector of A
    u_star - the actual top left eigenvector of A
    num_iter - the number of iterations of Power to do
    seed - for repeatable tests
    p - reduction percentage: A_reduced is (nx(1-p)*m)
    step_size - how often to regenerate A

    RETURN: xs - a list of iterations [0, 1, 2, ..., num_iter - 1]
            ys - the list of residuals per iteration
    Measure convergence of Power with some random subset of the columns of A
    """

    d = jl.percent_reduce(A.shape[1], p)

    xs, ys, label = subset_pow_swap(A, u_0, u_star, num_iter, seed, d, step_size)

    return xs, ys, f"{p}% {label}"