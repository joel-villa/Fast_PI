"""
A Johsnon Lindenstrauss implementation based on Section 5.2 of Dr. Saia's 
lecture on JL Projection:
https://www.cs.unm.edu/~saia/classes/506-s26/lec/HighDim+JLProjection.pdf
"""

import math
import numpy as np

import src.util.scikit_jl as util

def jl_entry_vals(d):
    """ Get a tuple of those possible values that can be put in the JL matrix

    Args: 
        d: reduced dimension
        
    Return: [-sqrt(1/n), sqrt(1/n)]
    """    
    lower = -1 * math.sqrt(1 / d)
    upper = math.sqrt(1 / d)
    return (lower,upper)

def jl_matrix(n, d, seed, eps):
    """ Randomly generate a JL matrix of dimensions (dxn)

    Args: 
        n: first dimension of reduction matrix
        d: second dimension of reduction matrix
        seed: for reproducible randomness
        eps: for jl validity checking

    Return: JL reduction matrix
    """
    if d is None:
        d = util.get_min_dim(X=np.zeros((n, n)), eps=eps)

    rng = np.random.default_rng(seed)
    return rng.choice(jl_entry_vals(d), size=(n, d))

def jl_simple(X, d, seed, eps):
    """Reduce dimensions of A, via simple jl projection

    Args:
        X: original matrix (nxm)
        d: desired dimension
        seed: for repeatable randomness
        eps: allowable error

    RETURN: reduced X (nxd), i.e. less columns
    """
    d = util.pre_jl(X=X, d=d, eps=eps)

    _, n = X.shape

    X_reduced = X @ jl_matrix(n=n, d=d, seed=seed, eps=eps)

    return X_reduced

def jl_gauss_mat(n, d, seed):
    """ Randomly generate a JL matrix of dimensions (nxd)

    Args: 
        n: first dimension of reduction matrix
        d: second dimension of reduction matrix
        seed: for reproducible randomness

    Return: JL reduction matrix

    See Section 4.1 of:
    https://www.cs.unm.edu/~saia/classes/506-s24/lec/HighDim+JLProjection.pdf
    """
    #TODO: untested
    rng = np.random.default_rng(seed)
    G = rng.normal(0, 1, size=(n, d))
    return G * (1 / math.sqrt(n))

# def jl_gauss(X, d, seed, eps=0.9):
#     """Reduce dimensions of A, via simple jl projection

#     Args:
#         X: original matrix (mxn)
#         d: desired dimension
#         seed: for repeatable randomness
#         eps: allowable error

#     RETURN: reduced X (nxd), i.e. less columns
#     """
    
#     util.check_valid_dimensions(X)
#     util.check_safe(X, d, eps)

#     _, n = X.shape

#     X_reduced = X @ jl_gauss_mat(n=n, d=d, seed=seed)

#     return X_reduced