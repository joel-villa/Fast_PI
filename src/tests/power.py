"""
For testing the convegence of a JL enhanced power iteration method

All methods in this file return xs, ys, and labels (for easy plotting)
"""

from scipy.sparse.linalg import svds
from scipy.linalg import norm # 2-norm by default

import numpy as np

from ..util.eig_functs import euclidean_dist
from ..util.scikit_jl import jl_gaussian
from ..util.scikit_jl import jl_sparse
from ..util.scikit_jl import percent_reduce
from ..util.power import topsing
from ..util.power import v_from_u

def baseline_pow_convergence(A, u_0, u_star, num_iter, seed):
    """
    The baseline power convergence
    """

    v =  v_from_u(A, u_0)

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)

    ys[0] = euclidean_dist(u_0, u_star)
    xs[0] = 0

    for i in range(1, num_iter):
        # NOTE: using scikit-learn -> top left eig (u) is of significance
        u, _, v = topsing(v0=v,
                          A=A, 
                          maxiter=1)
        
        # v = v.flatten() # make v 1D rather than 2D: (x,) rather than (x,1)

        euc_dist = euclidean_dist(u, u_star)
        ys[i] = euc_dist
        xs[i] = i
    
    return xs, ys, f"standard power {A.shape}"

def jl_reduced_pow_convergence(A, u_0, u_star, num_iter, seed, d, type="jl_gaussian"):
    """
    Convergence of Power Iteration on a JL-dimensionally reduced version of A
    """

    

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)

    match type:
        case "jl_gaussian":
            reduct_funct = jl_gaussian
        case _:
            reduct_funct = jl_sparse

    reduced_A = reduct_funct(A, d=d, seed=seed, eps=0.99)

    v =  v_from_u(reduced_A, u_0)

    ys[0] = euclidean_dist(u_0, u_star)
    xs[0] = 0

    for i in range(1, num_iter):
        # NOTE: using scikit-learn -> top left eig (u) is of significance

        u, _, v = topsing(v0=v,
                          A=reduced_A, 
                          maxiter=1)
        
        # v = v.flatten() # make v 1D rather than 2D: (x,) rather than (x,1)

        euc_dist = euclidean_dist(u, u_star)
        ys[i] = euc_dist
        xs[i] = i

    return xs, ys, f"jl-reduced power {reduced_A.shape} ({type})"

def jl_percent_reduced(A, u_0, u_star, num_iter, seed, p, type):
    """
    Convergence of Power Iteration on a JL-dimensionally reduced version of A
    """

    d = percent_reduce(A.shape[1], p)
    xs, ys, _ = jl_reduced_pow_convergence(A, u_0, u_star, num_iter, seed, d, type)
    
    return xs, ys, f"{p}% jl-reduced power ({type})"

def multi_jl_pow(A, u_0, u_star, num_iter, seed, d, step_size, type="jl_gaussian"):
    """
    Swapping between multiple JL reductions, attempting to  "enhance" power 
    iteration
    """

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)

    match type:
        case "jl_gaussian":
            reduct_funct = jl_gaussian
        case _:
            reduct_funct = jl_sparse

    reduced_A = reduct_funct(A, d=d, seed=seed, eps=0.99)

    v =  v_from_u(reduced_A, u_0)

    ys[0] = euclidean_dist(u_0, u_star)
    xs[0] = 0

    for i in range(1, num_iter):
        if (i % step_size == 0):
            # Randomly regenerate A
            reduced_A = reduct_funct(A, d=d, seed=seed*i, eps=0.99)
            v = v_from_u(reduced_A, u)

        # NOTE: using scikit-learn -> top left eig (u) is of significance

        u, _, v = topsing(v0=v,
                          A=reduced_A, 
                          maxiter=1)
        
        # v = v.flatten() # make v 1D rather than 2D: (x,) rather than (x,1)

        euc_dist = euclidean_dist(u, u_star)
        ys[i] = euc_dist
        xs[i] = i

    return xs, ys, f"reduced to {reduced_A.shape}, swapping every {step_size} ({type})"

def multi_jl_p_reduce(A, u_0, u_star, num_iter, seed, p, step_size, type):
    """
    Swapping between multiple JL reductions, attempting to  "enhance" power 
    iteration (percentage based)
    """

    d = percent_reduce(A.shape[1], p)
    xs, ys, _ = multi_jl_pow(A, u_0, u_star, num_iter, seed, d, step_size, type)
    
    return xs, ys, f"reduced {p}%, swapping every {step_size} ({type})"

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
    
    