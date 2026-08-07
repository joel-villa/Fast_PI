"""A file for computing the metadata of theoretical significance of a given 
sparse matrix"""

import scipy
from scipy.sparse.linalg import norm


def get_c_k(A:scipy.sparse) -> tuple[float, float]:
    """Get the power-law-distribution coefficients corresponding to A's row
    magnitude distribution

    Args:
        A (scipy.sparse): The matrix under question

    Returns:
        tuple[float, float]: 
            c: y-intercept of power-law distribution
            k: slope of power-law distribution
    """
    raise NotImplementedError("get_c_k is not yet implemented")

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
        A:scipy.sparse
) -> dict[str, int | float]:
    """Get the metadata of theoretical signficance for the provided matrix

    Args:
        A (scipy.sparse): matrix under question

    Returns:
        dict: 
            "n": number of rows in A
            "c": y-intercept of power-law distribution
            "k": slope of power-law distribution
            "kappa": -3k - 1 (always positive)
            "op-norm": ||A||_2
    """
    n = A.shape[0]
    c, k = get_c_k(A)
    kappa = get_kappa(k)
    op_norm = get_op_norm(A)

    return {
        "n": n,
        "c": c,
        "k": k,
        "kappa": kappa,
        "op_norm": op_norm,
    }