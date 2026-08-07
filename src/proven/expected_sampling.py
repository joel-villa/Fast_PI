"""A file for generating the expected sampling of a matrix A, given the sampling
scheme which was proven to be theoretically perfomant"""

from .util.meta_data import get_n_c_k
from ..util.constants import THIRTY_TWO_BIT_PRECISION

def expected_sampling_ub(
    matrix: str,
) -> float:
    """ Get an upper bound on the expected sampling of the matrix, under the 
    assumption that the sampling scheme is to pick the i'th row with probability
    pi, and to scale the i'th row by 1/sqrt(pi).

    Args:
        matrix (str): The name of a Suite-Sparse matrix

    Returns:
        float: expected rows kept of that matrix
    """
    n, c, k = get_n_c_k(matrix_name=matrix)

    if (abs(k + 1) <= THIRTY_TWO_BIT_PRECISION):
        # k = -1 check
        raise ValueError(f"For {matrix}, k = {k}")
    
    return c * (1 + (((n ** (k + 1)) - 1) / (k + 1)))