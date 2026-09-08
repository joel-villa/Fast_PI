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

from .util.meta_data import get_eig_info, get_row_norms
from .util.BoundsType import BoundsType
from .util.comp_data import preprocess #TODO: delete, this is for testing
from ..util.proven import get_A_tilde #TODO: delete ^
from numpy.linalg import norm #TODO: delete ^

def get_epsilon(
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
    print(f"L: {log_term}")

    numerator = - log_term + sqrt(log_term * (log_term - 18 * var_proxy))
    denominator = 3 * lambda_max * lambda_max

    return sqrt(numerator / denominator)

def get_strict_variance(
        value_top:float,
        vector_top:np.ndarray,
        row_mags:np.ndarray,
) -> float:
    """The most strict variance proxy: a summation

    Args:
        value_top (float): The top eigenvalue of A
        vector_top (float): The top eigenvector of A
        row_mags (np.ndaray): The two norm of all the rows of A

    Returns:
        float: The variance proxy
    """
    """ TODO: working with potentially really small numbers here, ensure that 
    zero rounding is kept to a minimum"""

    assert np.any(row_mags != 0) #Uh-oh, division by zero #TODO is this correct logic? 

    vector_top_4 = vector_top ** 4
    value_top_4 = value_top ** 4

    inv_row_mags = 1 / row_mags
    values = (inv_row_mags - 1) * value_top_4
    values = values * vector_top_4 # Element wise multiplication

    return np.sum(values)

def get_var_proxy(
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
            value_top, vector_top = get_eig_info(mat_name)
            row_mags = get_row_norms(mat_name)
            return get_strict_variance(
                value_top=value_top,
                vector_top=vector_top,
                row_mags=row_mags,
            )
        case _:
            raise NotImplementedError(f"Unexpected type: {bounds_type}")

def get_epsilon_bounds(
        mat_name:str,
        delta: float,
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
    lambda_max = get_eig_info(mat_name)[0]

    print(f"lambda: {lambda_max}")
        
    var_proxy = get_var_proxy(
        mat_name=mat_name,
        bounds_type=bounds_type,
    )

    print(f"S: {var_proxy}")

    epsilon = get_epsilon(
        delta=delta,
        var_proxy=var_proxy,
        lambda_max=lambda_max
    )

    print(f"epsilon: {epsilon}")

    ub = (1 + epsilon) * lambda_max
    lb = (1 - epsilon) * lambda_max

    str_rep = f"w.p. at least {delta}, ||~Av|| falls in [{lb}, {ub}]"
    
    return epsilon, (lb, ub), str_rep


if __name__ == '__main__':
    """Main for testing purposes
    """
    mats = [
        "1138_bus",
        "494_bus",
        "Harvard500",
        "bcspwr06",
        "bcsstk07",
        "bcsstk08",
        "bcsstk19",
        "bcsstk34",
        "bcsstm07",
        "blckhole",
        "cage7",
        "can_229",
        "dwt_193",
        "eris1176",
        "ex2",
        "fs_541_1",
        "gre_1107",
        "gre_343",
        "hor_131",
        "lshp1561",
        "msc00726",
        "nasa1824",
        "nos3",
        "tomography",
    ]

    mats = sorted(mats) #Alphabetical order
    for mat in mats:
        print(mat)
        print(f"get_epsilon_bounds(): {get_epsilon_bounds(mat, 0.9, BoundsType.STRICT)[2]}")

        A, _ = preprocess(mat)
        v = get_eig_info(mat)[1]
        A_tilde = get_A_tilde(
            A=A, 
            gen_type="two-norm",
            sf_kwargs={'power':1},
            seed=7,
        )

        A_tilde_v = A_tilde @ v
        print(f"shapes... A:{A.shape}, v:{v.shape}, ~Av:{A_tilde_v.shape}")

        print(f"||~Av||: {norm(A_tilde_v)}")
        print("\n\n")