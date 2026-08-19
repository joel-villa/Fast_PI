"""The following file contains Bernstein Concentration Bound results for the 
quantity ||~Av||, where ~A is the samples version of A, and v is the normalized 
top eigenvector of A

Assumptions: 
(1) The maximum row magnitude is 1
(2) The row magnitudes WLOG are upperbounded by a powerlaw distribution i.e.
    ||a_i|| = c * i^-k, for some constants c, k > 0
(3) The variance proxy is positive
(4) A is PSD, i.e. ||Av|| = lambda 
"""

import scipy

import numpy as np

# Exponentiation and logging is e-based
from math import log, sqrt

from .util.BoundsType import BoundsType

def epsilon(
        delta: float,
        var_proxy:float,
        lambda_max:float,
) -> float:
    """Get the epsilon (an error rate) corresponding to the given metrics

    Args:
        delta (float): the probability of success (falling within epsilon error)
        var_proxy (float): Some measure of variance
        lambda_max (float): The maximum eigenvalue of the matrix TODO: should we assume PSD?

    Returns:
        float: epsilon corresponding with the provided values
    """
    log_term = log((1 - delta) / 2)

    numerator = - log_term + sqrt(log_term * (log_term - 18 * var_proxy))
    denominator = 3 * lambda_max * lambda_max

    return sqrt(numerator / denominator)

def strict_variance(
        A:scipy.sparse,
        value_top:float,
        vector_top:float,
        row_mags:np.ndarray,
) -> float:
    """The most strict variance proxy: a summation

    Args:
        A (scipy.sparse): The matrix in question
        value_top (float): The top eigenvalue of A
        vector_top (float): The top eigenvector of A
        row_mags (np.ndaray): The two norm of all the rows of A

    Returns:
        float: The variance proxy
    """
    raise NotImplementedError()

def var_proxy(
        mat_name:str,
        bounds_type: BoundsType,
) -> float:
    """Get the variance proxy associated with the given type

    Args:
        mat_name (str): SS matrix name
        bounds_type (BoundsType): Type of bounds

    Returns:
        float: The variance proxy
    """
    match bounds_type:
        case BoundsType.STRICT:
            raise NotImplementedError()
        case _:
            raise NotImplementedError(f"Unexpected type: {bounds_type}")

def epsilon_bounds(
        mat_name:str,
        prob_success: float,
        bounds_type: BoundsType,
) -> tuple[float, tuple[float, float], str]:
    """Get the bernstein based epsilon bounds for the matrix

    Args:
        mat_name (str): Suite Sparse matrix
        prob_success (float): probability of success 
        bounds_type (BoundsType): power or non power based?

    Returns:
        tuple[float, tuple[float, float], str]: [
        float: epsilon,
        tuple[float, float]: [lowerbound, upperbound]
        str: string representation of info
        ]
    """
    raise NotImplementedError()
    # n, _ = get_n_norm(mat_name)
    
    # var_proxy = var_from_type(
    #     bounds_type=bounds_type,
    #     mat_name=mat_name,
    #     n=n,
    # )
    
    # return valid_epsilon(
    #     delta=delta,
    #     n=n,
    #     A_norm=op_norm, 
    #     var_proxy=var_proxy,
    # )