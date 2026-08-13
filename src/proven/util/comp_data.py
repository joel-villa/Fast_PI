"""A file for computing the metadata of theoretical significance of a given 
sparse matrix"""

import math

import numpy as np

import scipy
from scipy.sparse.linalg import norm

from ..preprocess import preprocess

from ...util.row_norms import get_sorted_row_norms, get_power_law_coefficients

N_STRING:str = "n"
C_STRING:str = "c"
K_STRING:str = "k"
KAPPA_STRING:str = "kappa"
NORM_STRING:str = "op_norm"
VAR_STRING:str = "var_proxy"
SCALE_STRING:str = "scale_factor"

META_DATA:set[str] = { 
    N_STRING,
    C_STRING, 
    K_STRING, 
    KAPPA_STRING, 
    NORM_STRING,
    VAR_STRING,
    SCALE_STRING,
}


def get_c_k(mat_name:str) -> tuple[float, float]:
    """Get the power-law-distribution coefficients corresponding to A's row
    magnitude distribution

    Args:
        mat_name (str): The matrix under question

    Returns:
        tuple[float, float]: 
            c: y-intercept of power-law distribution
            k: slope of power-law distribution
    """
    row_norms = get_sorted_row_norms(mat_name=mat_name, rescale=True)

    return get_power_law_coefficients(ys=row_norms)
    

def get_kappa(k:float) -> float:
    """Get kappa from k 

    Args:
        k (float): the slope of the power-law distribution on a log-log plot

    Returns:
        float: -3k - 1
    """
    return -3*k - 1

def get_op_norm(A:scipy.sparse) -> float:
    """Get the operator norm of the sparse matrix

    Args:
        A (scipy.sparse): the matrix

    Returns:
        float: ||A||_2
    """
    return norm(A, ord=2)

def get_var_proxy(A:scipy.sparse) -> float:
    """The variance proxy from Matrix Bernstein's

    Args:
        A (scipy.sparse): The matrix in question

    Returns:
        float: || sum_{i=1}^n (||a_i|| - ||a_i||^2)a_i^Ta_i||
    """

    sum_mat = np.zeros(shape=A.shape)

    for i in range(A.shape[0]):
        # Loop through the rows
        row_i = A[i, :] 

        # Compute row magnitudes
        i_vals = row_i.data
        norm_squared = np.sum(i_vals * i_vals) # sum of the square of the values
        norm_i = math.sqrt(norm_squared)

        if norm_i > 1.000001:
            # All rows should have row magnitudes less than or equal to 1
            raise ValueError(f"i'th row magnitude is {norm_i}, {norm_i} > 1")

        scalar_factor = norm_i * (1 - norm_i) 
        outer_prod = row_i.T @ row_i

        sum_mat += scalar_factor * outer_prod

    var_proxy = np.linalg.norm(sum_mat, ord=2)

    if var_proxy > A.shape[0] * 0.25000001:
        # something went wrong: x-x^2 <= 0.25
        raise ValueError(f"var_proxy = {var_proxy}")

    return float(var_proxy)

        

def get_metadata(
        mat_name: str,
) -> dict[str, int | float]:
    """Get the metadata of theoretical signficance for the provided matrix

    Args:
        mat_name (str): the matrix name (suite-sparse)
        A (scipy.sparse): matrix under question

    Returns:
        dict[str, int | float]: 
            "n": number of rows in A
            "c": y-intercept of power-law distribution
            "k": slope of power-law distribution
            "kappa": -3k - 1 (always positive)
            "op_norm": ||A||_2
            "var_proxy": variance proxy from matrix bernsteins
            "scale_factor": how much the matrix is scaled to get ||a_i|| <= 1
    """
    A, scale_factor = preprocess(mat_name)
    n = A.shape[0]
    c, k = get_c_k(mat_name)
    kappa = get_kappa(k)
    op_norm = get_op_norm(A)
    var_proxy = get_var_proxy(A)

    return {
        N_STRING: n,
        C_STRING: c,
        K_STRING: k,
        KAPPA_STRING: kappa,
        NORM_STRING: op_norm,
        VAR_STRING: var_proxy,
        SCALE_STRING: scale_factor,
    }