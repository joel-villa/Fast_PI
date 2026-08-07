"""A file for computing the metadata of theoretical significance of a given 
sparse matrix"""

import scipy
from scipy.sparse.linalg import norm

from Sparsification_Research.src.SSGetter import SSGetter

from ...util.row_norms import get_sorted_row_norms, get_power_law_coefficients


def preprocess(mat_name:str) -> tuple[scipy.sparse, float]:
    """Get A scaled by a factor of 1/max_i(||a_i||)

    Args:
        mat_name (str): the name of the suite-spasrse matrix

    Returns:
        tuple[scipy.sparse, float]: 
            scipy.sparse: the matrix in sparse format
            float: the scaling factor
    """
    ssgetter = SSGetter()
    A = ssgetter.get(mat_name)
    
    max_row_magnitude = get_sorted_row_norms(mat_name, rescale=False)[0]

    # Rescale A
    A = A / max_row_magnitude

    return A, max_row_magnitude
    

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
            "op-norm": ||A||_2
    """
    A, scale_factor = preprocess(mat_name)
    n = A.shape[0]
    c, k = get_c_k(mat_name)
    kappa = get_kappa(k)
    op_norm = get_op_norm(A)

    return {
        "n": n,
        "c": c,
        "k": k,
        "kappa": kappa,
        "op_norm": op_norm,
        "scale_factor": scale_factor,
    }