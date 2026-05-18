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
    
    

def topsing(v0, A, maxiter=10, tolerance=1e-07):
    """
    v0      - an initial guess for the top right eigenvector (m-dimensional)
    A       - A matrix (nxm) s.t. n is less than or equal to m (for relative 
              efficiency)
    maxiter - how many iterations of power? 
            - maxiter = -1 -> run until tolerance met

    RETURN: u - top left eigenvector approximation (n-dimensional)
            s - singular value (akin to eigenvalue)
            v - top right eigenvector approximation (m-dimensional)
    Adapted from section "4.4.2. Computing the top singular vector", found here:
    https://mmids-textbook.github.io/chap04_svd/04_power/roch-mmids-svd-power.html

    For convergence checking, used this resource:
    https://www.geeksforgeeks.org/python/power-method-determine-largest-eigenvalue-and-eigenvector-in-python/ 
    """
    x = v0.copy()
    B = A.T @ A 

    # Normalize initial vector (good practice)
    x = x / np.linalg.norm(x)

    # top singular value is None for first iteration
    s_prev = None

    # Initialize v and s
    v = None
    s = None

    for _ in range(maxiter):
        x = B @ x

        # compute top left 
        v = x / norm(x)

        # top singular value
        s = norm(A @ v)

        # Check convergence
        if s_prev is not None and abs(s - s_prev) < tolerance:
            # print(i)
            break
        s_prev = s

    u = A @ v / s
    return u, s, v