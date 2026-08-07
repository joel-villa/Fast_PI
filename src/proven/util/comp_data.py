"""A file for computing the metadata of theoretical significance of a given 
sparse matrix"""

import scipy

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

def get_metadata(A:scipy.sparse) -> tuple[int, float, float, float, float]:
    """Get the metadata of theoretical signficance for the provided matrix

    Args:
        A (scipy.sparse): matrix under question

    Returns:
        tuple[float, float, float, float, int]: 
            n: number of rows in A
            c: y-intercept of power-law distribution
            k: slope of power-law distribution
            kappa: -3k - 1 (always positive)
            op-norm: ||A||_2
    """
    n = A.shape[0]
    c, k = get_c_k(A)
    kappa = get_kappa(k)
    op_norm = get_op_norm(A)

    return n, c, k, kappa, op_norm