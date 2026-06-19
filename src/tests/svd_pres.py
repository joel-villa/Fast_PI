"""
A file containing tests for singular vector preservation 

Functions in this file have Return type as follows:
    xs: an array of reduction percentages (ps)
    ys: a 2d array of relative error in top k left singular vectors
"""
import numpy as np

from scipy.sparse.linalg import svds

from ..util.subset import p_reduce_A
from ..util.sparsify import p_sparse


def eig_pres_left(A_tilde, u0, u_star, k, seed):
    """ Test the preservation of A's top-left eigenvectors
    
    Args:
        A_tilde: The modified version of A
        u0: Initial guess for top left eigenvectors (TODO:needs to be converted to v0)
        u_star: The top-k left singular vectors of the original matrix
        k: The number of singular vectors to compute
        seed: for repeated randomness
    Return: 
        2-norm difference between A_tilde's top-left singular vector's and A's
        top left singular vectors
    """

    u, s, vh = svds(
        A=A_tilde,
        k=k,
        # maxiter=256, #TODO: do we need this? 
        rng=seed,
    )

    # Difference from actual
    diff1 = u - u_star
    diff2 = u + u_star

    # 2-norm of difference (left -> columns)
    norm1 = np.linalg.norm(diff1, ord=2, axis=0)
    norm2 = np.linalg.norm(diff2, ord=2, axis=0)
    norm_u = np.linalg.norm(u_star, ord=2, axis=0)

    # Take minimum 2-norm difference per column
    eig_perturbance = np.minimum(norm1, norm2)

    # Error relative to the magnitude of the solution
    relative_error = eig_perturbance / norm_u

    relative_error = relative_error.flatten()

    if relative_error.shape[0] != k:
        raise ValueError(f"{relative_error.shape} != {k} = k")
    
    return relative_error


def eig_pres_right(A_tilde, v0, v_star, k, seed):
    """ Test the preservation of A's top-right eigenvectors
    
    Args:
        A_tilde: The modified version of A
        v0: Initial guess for top right eigenvectors
        v_star: The top-k right singular vectors of the original matrix
        k: The number of singular vectors to compute
        seed: for repeated randomness
    Return: 
        2-norm difference between A_tilde's top-right singular vector's and A's
        top-right singular vectors
    """

    u, s, vh = svds(
        A=A_tilde,
        k=k,
        # maxiter=256, #TODO: do we need this? 
        rng=seed,
    )

    # Difference from actual
    diff1 = vh - v_star
    diff2 = vh + v_star

    # 2-norm of difference (right -> rows)
    norm1  = np.linalg.norm( diff1, ord=2, axis=1)
    norm2  = np.linalg.norm( diff2, ord=2, axis=1)
    norm_v = np.linalg.norm(v_star, ord=2, axis=1)

    # Take minimum 2-norm difference per column
    eig_perturbance = np.minimum(norm1, norm2)

    # Error relative to the magnitude of the solution
    relative_error = eig_perturbance / norm_v

    relative_error = relative_error.flatten()

    if relative_error.shape[0] != k:
        raise ValueError(f"{relative_error.shape} != {k} = k")
    
    return relative_error

def subset_p(A, u_0, u_star, k, seed, ps, type, gamma):
    """ Test of random col sampling based on percent of reduction

    Args:
        A: the matrix to do power iteration on
        u_0: initial guess for top left singular vectors
        u_star: The top-k left singular vectors of the original matrix
        k: The number of singular vectors to compute
        seed: for repeated randomness
        ps: reduction percentages
        type: string representation of reduction type
        gamma: gamma parameter for the gamma-ridge leverage score
    Return:
        xs: an array of reduction percentages (ps)
        ys: a 2d array of relative error in top k left singular vectors 
            (each column represents a distinct k-value)
    """
    ys_shape = (ps.shape[0], k)
    ys = np.zeros(ys_shape)

    for i, p in enumerate(ps):
        A_reduced = p_reduce_A(
            A=A, 
            p=p,
            seed=seed,
            type=type,
            gamma=gamma,
        )
        ys_i = eig_pres_left(
            A_tilde=A_reduced,
            u0=u_0,
            u_star=u_star,
            k=k,
            seed=seed,
        )
        ys[i] = ys_i
    
    return ps, ys

def sparse_p(A, u_0, u_star, k, seed, ps, type):
    """ Preservation of top k eigenvectors of a sparsified version of A

    Args: 
        A: the original matrix
        u_0: initial guess for top left singular vectors of A
        u_star: The top-k left singular vectors of the original matrix
        k: The number of singular vectors to compute
        seed: for repeated randomness
        ps: reduction percentages
        type: string representation of sparsification type

    Returns:
        xs: an array of reduction percentages (ps)
        ys: a 2d array of relative error in top k left singular vectors 
            (each column represents a distinct k-value)
    """
    ys_shape = (ps.shape[0], k)
    ys = np.zeros(ys_shape)

    for i, p in enumerate(ps):
        A_sparse = p_sparse(
            A=A, 
            p=p,
            seed=seed,
            type=type,
        )
        ys_i = eig_pres_left(
            A_tilde=A_sparse,
            u0=u_0,
            u_star=u_star,
            k=k,
            seed=seed,
        )
        ys[i] = ys_i
    
    return ps, ys
