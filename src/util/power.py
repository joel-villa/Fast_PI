"""
For Power Iteration helper functions
"""

import numpy as np
from scipy.linalg import norm # 2-norm by default

def s_from_u(A, u):
    """Get the 'score' of the left vector of A

    Args:
        A: the matrix in question
        u: the vector in question

    Return:
        singular value associated with the two
    """
    s = norm(A.T @ u)
    return s

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

    s = s_from_u(A, u)
    
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

    for _ in range(maxiter):
        x = A.T @ (A @ x)
    v = x / norm(x)
    s = norm(A @ v)
    u = A @ v / s
    return u, s, v

def count_mults(A, maxiter=10):
    """
    v0 - some vector with same dimension as A's top right eigenvectors
    A  - the matrix to do power iteration on
    maxiter - number of iterations of Power Iteration

    RETURN: total number of scalar mults

    Calculate the number of matrix multiplications for a set number of 
    iterations of Power Iteration
    """
    m, n = A.shape

    # Cost of A @ x: A is (mxn), x is (nx1)
    cost_Ax = m * n * 1

    # Cost of A.T @ (A @ x): A.T is nxm and A @ x is mx1
    cost_ATAx = n * m * 1

    # Number of scalar mults for a single iteration
    scalar_mults = cost_Ax + cost_ATAx

    # Number of scalar mults of maxiter iterations
    scalar_mults *= maxiter

    return scalar_mults

    

def power_lazy(v0, A, P, maxiter=10):
    """ A version of JL-enhanced power iteration, which avoids mat-mat 
    multiplication, instead doing only matrix vector mults

    Args: 
        v0: an initial guess for the top right eigenvector (m-dimensional)
        A: A matrix (nxm) s.t. n is less than or equal to m (for relative 
           efficiency)
        P: Projection matrix
        maxiter: how many iterations of Power? 

    RETURN: u - top left eigenvector approximation (n-dimensional)
            s - singular value (akin to eigenvalue)
            v - top right eigenvector approximation (m-dimensional)

    Adapted from section "4.4.2. Computing the top singular vector", found here:
    https://mmids-textbook.github.io/chap04_svd/04_power/roch-mmids-svd-power.html
    """
    x = v0.copy()

    for _ in range(maxiter):
        x = P.T @ (A.T @ (A @ (P @ x))) # x is dx1
    v = x / norm(x)
    s = norm(A @ (P @ v))
    u = A @ (P @ v) / s
    return u, s, v

def count_mults_lazy(A, P, maxiter=10):
    """
    v0 - some vector with same dimension as A's top right eigenvectors
    A  - the matrix to do power iteration on
    maxiter - number of iterations of Power Iteration

    RETURN: total number of scalar mults

    Calculate the number of matrix multiplications for a set number of 
    iterations of Power Iteration
    """
    m, n = A.shape
    n, d = P.shape

    cost_Px = n * d * 1

    cost_APx = m * n * 1

    cost_ATAPx = n * m * 1

    cost_PTATAPx = d * n * 1

    # Number of scalar mults for a single iteration
    single_iter_cost = cost_Px + cost_APx + cost_ATAPx + cost_PTATAPx

    # Number of scalar mults of maxiter iterations
    scalar_mults = maxiter * single_iter_cost

    return scalar_mults


# def topsing(v0, A, maxiter=10, tolerance=1e-07):
#     """
#     v0      - an initial guess for the top right eigenvector (m-dimensional)
#     A       - A matrix (nxm) s.t. n is less than or equal to m (for relative 
#               efficiency)
#     maxiter - how many iterations of power? 
#             - maxiter = -1 -> run until tolerance met

#     RETURN: u - top left eigenvector approximation (n-dimensional)
#             s - singular value (akin to eigenvalue)
#             v - top right eigenvector approximation (m-dimensional)
#     Adapted from section "4.4.2. Computing the top singular vector", found here:
#     https://mmids-textbook.github.io/chap04_svd/04_power/roch-mmids-svd-power.html

#     For convergence checking, used this resource:
#     https://www.geeksforgeeks.org/python/power-method-determine-largest-eigenvalue-and-eigenvector-in-python/ 
#     """
#     x = v0.copy()
#     B = A.T @ A 

#     # Normalize initial vector (good practice)
#     x = x / np.linalg.norm(x)

#     # top singular value is None for first iteration
#     s_prev = None

#     # Initialize v and s
#     v = None
#     s = None

#     for _ in range(maxiter):
#         x = B @ x

#         # compute top left 
#         v = x / norm(x)

#         # top singular value
#         s = norm(A @ v)

#         # Check convergence
#         if s_prev is not None and abs(s - s_prev) < tolerance:
#             # print(i)
#             break
#         s_prev = s

#     u = A @ v / s
#     return u, s, v