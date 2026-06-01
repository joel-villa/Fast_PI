"""
Some tests for dimensionality reduction where you use random subset of the 
columns to see if power iteration converges faster
"""
import numpy as np
from ..util import power as pwr
from ..util import eig_functs as eig
from ..util import scikit_jl as jl
from ..util import subset as sub

#TODO: something similar to Chan's (randomly pick larger and larger sample-sizes)

def subset_pow(A, u_0, u_star, num_iter, seed, d, type):
    """Measure convergence of Power with some random subset of the columns of A

    Args: 
        A: the matrix (nxm)
        u_0: an initial guess for the top left eigenvector of A
        u_star: the actual top left eigenvector of A
        num_iter: the number of iterations of power to do
        seed: for repeatable tests
        d: reduction size: A_reduced is (nxd)
        type: the type of reduction to do

    Return: 
        xs: a list of iterations [0, 1, 2, ..., num_iter - 1]
        ys: the list of residuals per iteration
    """

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)

    ys[0] = eig.euclidean_dist(u_0, u_star)
    xs[0] = 0

    sub.reduce_A(A=A, d=d, seed=seed, type="col_random", gamma=None)

    A_reduced = sub.select_d_random_columns(A, d, seed)

    v =  pwr.v_from_u(A_reduced, u_0)

    for i in range(1, num_iter):
        # NOTE: using scikit-learn -> top left eig (u) is of significance

        u, _, v = pwr.topsing(v0=v,
                          A=A_reduced, 
                          maxiter=1)
        
        euc_dist = eig.euclidean_dist(u, u_star)
        ys[i] = euc_dist
        xs[i] = i
    
    num_mults = pwr.count_mults(v0=v, A=A_reduced, maxiter=num_iter - 1)

    return xs, ys, f"column subset {A_reduced.shape}; {num_mults:,} mults"

def percent_subset_pow(A, u_0, u_star, num_iter, seed, p, type):
    """Measure convergence of Power with some random subset of the columns of A

    Args: 
        A: the matrix (nxm)
        u_0: an initial guess for the top left eigenvector of A
        u_star: the actual top left eigenvector of A
        num_iter: the number of iterations of power to do
        seed: for repeatable tests
        p: reduction percentage
        type: the type of reduction to do

    Return: 
        xs: a list of iterations [0, 1, 2, ..., num_iter - 1]
        ys: the list of residuals per iteration
    """

    d = jl.percent_reduce(A.shape[1], p)

    xs, ys, label = subset_pow(A, u_0, u_star, num_iter, seed, d, type)

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

    A_reduced = sub.select_d_random_columns(A, d, seed)

    v =  pwr.v_from_u(A_reduced, u_0)

    for i in range(1, num_iter):
        # NOTE: using scikit-learn -> top left eig (u) is of significance
        if (i % step_size == 0):
            # Randomly regenerate A
            reduced_A = sub.select_d_random_columns(A, d=d, seed=seed*i)
            v = pwr.v_from_u(reduced_A, u)
            # TODO: not currently counting number of mults in v_from_u() call

        u, _, v = pwr.topsing(v0=v,
                          A=A_reduced, 
                          maxiter=1)
        
        euc_dist = eig.euclidean_dist(u, u_star)
        ys[i] = euc_dist
        xs[i] = i

    num_mults = pwr.count_mults(v0=v, A=A_reduced, maxiter=num_iter - 1)

    return xs, ys, f"column subset {A_reduced.shape} swapping every {step_size}; {num_mults:,} mults"

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