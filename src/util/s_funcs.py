"""
For some scale distribution generating functions

All functions in this file have the following definition

Args:
    A: a CSR sparse matrix (mxn)
    ...
Return: a numpy array of length m, which gives the probability of selecting
        any of the m rows of A, indendently, s.t. all elements of the array are 
        in the interval [-1, 1]
"""

import numpy as np

from scipy.sparse.linalg import norm

def uniform(A, s):
    """ Probability of selecting i'th column is uniform
    
    Args:
        A: a CSR sparse matrix (mxn)
        p: the amount to scale the i'th column [-1, 1]
    Return: a numpy array of length m, which gives the probability of selecting
            any of the m rows of A, indendently
    """

    if (s < -1 or s > 1):
        raise ValueError(f"Uh-uh, s needs to be in range [-1, 1], not: {s}")

    return np.full(A.shape[0], s)

def norm_max_based(A, norm_ord):
    """ Probability of selecting i'th column is dependent on the norm of A, 
    se
    
    Args:
        A: a CSR sparse matrix (mxn)
    Return: a numpy array of length m, which gives the probability of selecting
            any of the m rows of A, indendently
    """
    row_norms = norm(A, ord=norm_ord, axis=1)
    max_row_norm = row_norms.max()

    # Divide all by the maximum row_norm
    scales = row_norms / max_row_norm

    return scales
