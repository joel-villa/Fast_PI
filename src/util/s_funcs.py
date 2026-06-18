"""
For some scale distribution generating functions

All functions in this file have the following definition

Args:
    A: a CSR sparse matrix (mxn)
    ...
Return: a numpy array of length m, which gives the probability of selecting
        any of the m rows of A, indendently
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

def two_norm(A):
    """ Probability of selecting i'th column is dependent on the two-norm of A
    
    Args:
        A: a CSR sparse matrix (mxn)
    Return: a numpy array of length m, which gives the probability of selecting
            any of the m rows of A, indendently
    """
    row_norms = norm(A, ord=2, axis=1)
    fro_norm = norm(A, ord="fro")

    # The sqrt of the probabiility of keeping each row
    # (NOTE: numpy `*` operator and `/` are element-wise)
    scales = row_norms / fro_norm

    return scales
