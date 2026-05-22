"""
A Johsnon Lindenstrauss implementation based on Section 5.2 of Dr. Saia's 
lecture on JL Projection:
https://www.cs.unm.edu/~saia/classes/506-s26/lec/HighDim+JLProjection.pdf
"""

import math
import numpy as np

import scikit_jl as util

def jl_entry_vals(d):
    """ Get a tuple of those possible values that can be put in the JL matrix

    Args: 
        d: reduced dimension
        
    Return: [-sqrt(1/n), sqrt(1/n)]
    """    
    lower = -1 * math.sqrt(1 / d)
    upper = math.sqrt(1 / d)
    return (lower,upper)

def jl_matrix(n, d, seed):
    """ Randomly generate a JL matrix of dimensions (dxn)

    Args: 
        n: first dimension of reduction matrix
        d: second dimension of reduction matrix
        seed: for reproducible randomness

    Return: JL reduction matrix
    """
    rng = np.random.default_rng(seed)
    return rng.choice(jl_entry_vals(d), size=(n, d))

def jl_simple(A, d, seed, eps=0.9):
    """Reduce dimensions of A, via simple jl projection

    Args:
        A: original matrix (nxm)
        d: desired dimension
        seed: for repeatable randomness
        eps: allowable error

    RETURN: reduced A (nxd), i.e. less columns
    """
    
    util.check_valid_dimensions(A)
    util.check_safe(A, d, eps)

    _, n = A.shape

    A_reduced = A @ jl_matrix(n=n, d=d, seed=seed)

    return A_reduced