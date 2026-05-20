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
    """
    The baseline power convergence
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
    
    return xs, ys, f"standard power {A.shape}; {pwr.count_mults(A, num_iter):,} mults"

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
    scalar_mults += pwr.count_mults(reduced_A, num_iter)

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

    num_mults += pwr.count_mults(A=reduced_A, maxiter=num_iter)

    return xs, ys, f"{type} {reduced_A.shape} swapping every {step_size}; {num_mults:,} mults"

def multi_jl_p_reduce(A, u_0, u_star, num_iter, seed, p, step_size, type):
    """
    Swapping between multiple JL reductions, attempting to  "enhance" power 
    iteration (percentage based)
    """

    d = jl.percent_reduce(A.shape[1], p)
    xs, ys, lbl = multi_jl_pow(A, u_0, u_star, num_iter, seed, d, step_size, type)
    
    return xs, ys, f"{p}% {lbl}"

# def svds_convergence(A, v0, v_star, num_iter, seed):
#     """
#     The baseline SVD convergence
#     NOTE: svds() from scipy.sparse.linalg is not an itterative method
#     """
#     v = v0.copy()

#     xs = np.zeros(num_iter)
#     ys = np.zeros(num_iter)

#     for i in range(num_iter):
#         # NOTE: using scikit-learn -> top left eig is of significance
#         v, _, _ = svds(A, 
#                        k=1,        # top eigenvector
#                        which='LM', # top eigenvector
#                        v0=v, #TODO
#                        maxiter=1, 
#                        random_state=seed) #TODO should this be more random (ie: add i to seed)
        
#         v = v.flatten() # make v 1D rather than 2D: (x,) rather than (x,1)

#         euc_dist = euclidean_dist(v, v_star)
#         ys[i] = euc_dist
#         xs[i] = i

#     # print(f"xs: {xs}")
#     # print(f"ys: {ys}")
    
#     return xs, ys, f"standard svd"
    
    