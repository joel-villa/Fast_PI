"""
For testing the convegence of a JL enhanced power iteration method

All methods in this file return xs, ys, and labels (for easy plotting)
"""

from scipy.sparse.linalg import svds
from scipy.linalg import norm # 2-norm by default

import numpy as np

from ..util import eig_functs as eig
from ..util import scikit_jl as jl
from ..util import power as pwr

def baseline_pow_convergence(A, u_0, u_star, num_iter, seed):
    """ The baseline power convergence (terminates based on num_iter)

    Args:
        A: the matrix to do power iteration on
        u_0: initial guess for top left eigenvector
        u_star: actual top left eigenvector
        num_iter: number of iterations to do power
        seed: for repeatable randomness (doesn't work due to scikit's not 
              supporting this functionality)
    Return:
        xs: an array [0, 1, 2, ..., num_iter - 1]
        ys: an array of residual of each guess for u (top left eigencector)
        lbl: the string representation of this test
    """

    v =  pwr.v_from_u(A, u_0)

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)

    ys[0] = eig.euclidean_dist(u_0, u_star)
    xs[0] = 0

    for i in range(1, num_iter):
        # NOTE: using scikit-learn -> top left eig (u) is of significance
        u, _, v = pwr.topsing(v0=v,
                          A=A, 
                          maxiter=1)
        
        # v = v.flatten() # make v 1D rather than 2D: (x,) rather than (x,1)

        euc_dist = eig.euclidean_dist(u, u_star)
        ys[i] = euc_dist
        xs[i] = i
    
    return xs, ys, f"standard power {A.shape}; {pwr.count_mults(A, num_iter - 1):,} mults"


def baseline_pwr_tolerance_termination(A, u_0, u_star, num_iter, seed, tol=1e-07):
    """ The baseline power convergence (terminates based on tolerance)

    Tolerance termination based on:
        https://www.geeksforgeeks.org/python/power-method-determine-largest-eigenvalue-and-eigenvector-in-python/

    Args:
        A: the matrix to do power iteration on
        u_0: initial guess for top left eigenvector
        u_star: actual top left eigenvector
        num_iter: max number of iterations to do power
        seed: for repeatable randomness (doesn't work due to scikit's not 
              supporting this functionality)
        tol: How accurate should the top singular value be? 
    Return:
        xs: an array [0, 1, 2, ..., num_iter - 1]
        ys: an array of residual of each guess for u (top left eigencector)
        lbl: the string representation of this test
    """

    v =  pwr.v_from_u(A, u_0)

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)

    ys[0] = eig.euclidean_dist(u_0, u_star)
    xs[0] = 0

    # top singular value is None for first iteration
    s_prev = None

    for i in range(1, num_iter):
        # NOTE: using scikit-learn -> top left eig (u) is of significance
        u, s_curr, v = pwr.topsing(v0=v,
                          A=A, 
                          maxiter=1)
        
        if s_prev is not None and abs(s_curr - s_prev) < tol:
            # Converged
            break

        euc_dist = eig.euclidean_dist(u, u_star)
        ys[i] = euc_dist
        xs[i] = i

        s_prev = s_curr
    
    return xs, ys, f"standard power {A.shape}; {pwr.count_mults(A, num_iter - 1):,} mults"

def jl_reduced_pow_convergence(A, u_0, u_star, num_iter, seed, d, type="jl_gaussian"):
    """
    Convergence of Power Iteration on a JL-dimensionally reduced version of A
    """

    # tracking number of scalar mutls
    scalar_mults = 0

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)

    match type:
        case "jl_gaussian":
            reduct_funct = jl.jl_gaussian
        case _:
            reduct_funct = jl.jl_sparse

    reduced_A = reduct_funct(A, d=d, seed=seed, eps=0.99)

    scalar_mults += jl.reduction_cost(A, d=d)

    v =  pwr.v_from_u(reduced_A, u_0)

    ys[0] = eig.euclidean_dist(u_0, u_star)
    xs[0] = 0

    for i in range(1, num_iter):
        # NOTE: using scikit-learn -> top left eig (u) is of significance

        u, _, v = pwr.topsing(v0=v,
                          A=reduced_A, 
                          maxiter=1)
        
        # v = v.flatten() # make v 1D rather than 2D: (x,) rather than (x,1)

        euc_dist = eig.euclidean_dist(u, u_star)
        ys[i] = euc_dist
        xs[i] = i

    # Number of scalar mults for power
    scalar_mults += pwr.count_mults(reduced_A, num_iter - 1)

    return xs, ys, f"{type} {reduced_A.shape}; {scalar_mults:,} mults"

def jl_percent_reduced(A, u_0, u_star, num_iter, seed, p, type):
    """
    Convergence of Power Iteration on a JL-dimensionally reduced version of A
    """

    d = jl.percent_reduce(A.shape[1], p)
    xs, ys, lbl = jl_reduced_pow_convergence(A, u_0, u_star, num_iter, seed, d, type)
    
    return xs, ys, f"{p}% {lbl}"

def multi_jl_pow(A, u_0, u_star, num_iter, seed, d, step_size, type="jl_gaussian"):
    """
    Swapping between multiple JL reductions, attempting to  "enhance" power 
    iteration
    """

    # Track number of scalar mults
    num_mults = 0 

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)

    match type:
        case "jl_gaussian":
            reduct_funct = jl.jl_gaussian
        case _:
            reduct_funct = jl.jl_sparse

    reduced_A = reduct_funct(A, d=d, seed=seed, eps=0.99)

    num_mults += jl.reduction_cost(X=A, d=d)

    v =  pwr.v_from_u(reduced_A, u_0)

    ys[0] = eig.euclidean_dist(u_0, u_star)
    xs[0] = 0

    for i in range(1, num_iter):
        if (i % step_size == 0):
            # Randomly regenerate A
            reduced_A = reduct_funct(A, d=d, seed=seed*i, eps=0.99)
            v = pwr.v_from_u(reduced_A, u) #TODO: count this in number of scalar mults? 
            num_mults += jl.reduction_cost(X=A, d=d)

        # NOTE: using scikit-learn -> top left eig (u) is of significance

        u, _, v = pwr.topsing(v0=v,
                          A=reduced_A, 
                          maxiter=1)
        
        # v = v.flatten() # make v 1D rather than 2D: (x,) rather than (x,1)

        euc_dist = eig.euclidean_dist(u, u_star)
        ys[i] = euc_dist
        xs[i] = i

    num_mults += pwr.count_mults(A=reduced_A, maxiter=num_iter - 1)

    return xs, ys, f"{type} {reduced_A.shape} swapping every {step_size}; {num_mults:,} mults"

def multi_jl_p_reduce(A, u_0, u_star, num_iter, seed, p, step_size, type):
    """
    Swapping between multiple JL reductions, attempting to  "enhance" power 
    iteration (percentage based)
    """

    d = jl.percent_reduce(A.shape[1], p)
    xs, ys, lbl = multi_jl_pow(A, u_0, u_star, num_iter, seed, d, step_size, type)
    
    return xs, ys, f"{p}% {lbl}"