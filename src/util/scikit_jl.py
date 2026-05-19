"""
For interfacing with scikit learn's JL reduction
"""

import numpy as np
from sklearn import random_projection
from math import ceil 

def percent_reduce(n, p):
     """
     n - an integer (orignal dimension)
     p - the percent to reduce n by

     Reduce dimension n by p percent
     """

     reduce_ammount = ceil(n * p * 0.01)
     d = n - reduce_ammount
     
     return d

def check_valid_dimensions(A):
    if (A.shape[0] > A.shape[1]):
        raise ValueError(f"Scikit JL requires rows < cols, but A has shape: {A.shape}")
    
    if (min(A.shape) <= 1):
            raise ValueError(f"Matrix too small: {A.shape}")
    
def jl_gaussian(X, d, seed, eps=0.9):
    """
    X - original matrix (nxm)
    d - desired dimension
    seed - for repeatable randomness
    eps - allowable error

    RETURN: reduced X (nxd), i.e. less columns

    Reduce dimensions of X, via scikit learn's gaussian method
    """
    
    check_valid_dimensions(X)

    transformer = random_projection.GaussianRandomProjection(n_components=d, eps=eps, random_state=seed)
    X_new = transformer.fit_transform(X)
    return X_new

def jl_sparse(X, d, seed, eps=0.9):
    """
    X - original matrix
    d - desired dimension
    seed - for repeatable randomness
    eps - allowable error

    RETURN: reduced X

    Reduce dimensions of X, via scikit learn's gaussian method
    """

    check_valid_dimensions(X)

    transformer = random_projection.SparseRandomProjection(n_components=d, eps=eps, random_state=seed)
    X_new = transformer.fit_transform(X)
    return X_new

def reduction_cost(X, d):
     """
     X - original matrix
     d - reduced dimension

     Get the number of scalar mults required for the given dimensionality 
     reduction

     Projection matrix is (n x d) to get reduced matrix of (m x d)
     """

     # X is an (m x n) matrix
     m, n = np.shape(X)

     # (m x n) matrix times (n x d) matrix is m*n*d scalar mults
     return m * n * d

     