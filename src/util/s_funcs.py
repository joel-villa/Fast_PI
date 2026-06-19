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

def norm_max_based(A, norm_ord, power):
    """ Probability of selecting i'th column is dependent on the norm of A, 
    se
    
    Args:
        A: a CSR sparse matrix (mxn)
        norm_ord: what order norm to use
        power: to what power to raise the scales, in range: [0, inf)
            increasing power -> scales closer to zero -> less rows chosen -> higher variance
            decreasing power -> scales closer to one -> more rows chosen -> lower variance
    Return: a numpy array of length m, which gives the probability of selecting
            any of the m rows of A, indendently
    """
    if power < 0:
        raise ValueError(f"power must be nonegative, {power} invalid")
    if power == 0:
        # All scales are one -> return a one-filled array of length m
        return np.full(A.shape[0], 1)
    
    row_norms = norm(A, ord=norm_ord, axis=1)
    max_row_norm = row_norms.max()

    # Divide all by the maximum row_norm
    scales = row_norms / max_row_norm

    # Raise the scales to the power
    scales = scales ** power

    return scales
