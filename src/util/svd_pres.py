"""
Helper Functions for SVD Preservation tests
"""
import numpy as np

from scipy.sparse.linalg import svds


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
        v0=v0,
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