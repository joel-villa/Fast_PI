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
    
def get_min_dim(X, eps):
    """Get the minimum number of dimensions allowed for a reduction in the 
    number of columns of the matrix X

    Args: 
        X: the matrix to be reduced in a JL fashion
        eps: the ammount of allowed error
    
    Return:
        minimum dimension allowed
    """
    num_cols = np.shape(X)[0]

    return random_projection.johnson_lindenstrauss_min_dim(num_cols, eps=eps)
    
def check_safe(X, d, eps):
    """ Verify d is a valid dimensionality reduction

    Args:
        X: matrix to reduce columns of via JL
        d: lower dimension
        eps: allowable error
    """
    min_d = get_min_dim(X, eps)
    
    if (d < min_d):
         print(f"WARNING: eps = {eps}; unsafe dimensionality reduction d = {d} < {min_d} = min_d ")
    
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
    check_safe(X, d, eps)

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
    check_safe(X, d, eps)

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

     