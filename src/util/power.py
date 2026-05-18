"""
For SVD functions
"""

import numpy as np
from scipy.linalg import norm # 2-norm by default

def v_from_u(A, u):
    """
    A - matrix (nxm) s.t. n >= m
    u - top left eigenvector of A (n-dimensional)

    RETURN: v - m-dimensional top right eigenvector of A

    Get the top right eigenvector of A, given the top left eigenvector of A
    NOTE: 
    A * v = s * u -> A^T * u = s * v
    """
    #TODO: DEBUG and use this for consistent initialization 
    # (i.e. every iteration starts w/ same residue)

    s = norm(A.T @ u)
    v = A.T @ u / s
    return v
    
    

def topsing(v0, A, maxiter=10):
    """
    v0      - an initial guess for the top right eigenvector (m-dimensional)
    A       - A matrix (nxm) s.t. n is less than or equal to m (for relative 
              efficiency)
    maxiter - how many iterations of SVD? 

    RETURN: u - top left eigenvector approximation (n-dimensional)
            s - singular value (akin to eigenvalue)
            v - top right eigenvector approximation (m-dimensional)
    Adapted from section "4.4.2. Computing the top singular vector", found here:
    https://mmids-textbook.github.io/chap04_svd/04_power/roch-mmids-svd-power.html
    """
    x = v0.copy()
    B = A.T @ A 
    for _ in range(maxiter):
        x = B @ x
    v = x / norm(x)
    s = norm(A @ v)
    u = A @ v / s
    return u, s, v