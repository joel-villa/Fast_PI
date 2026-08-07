"""Theoretical results are reliant on matrices being scaled s.t. the max row
magnitude is 1"""

import scipy

from Sparsification_Research.src.SSGetter import SSGetter

from ..util.row_norms import get_sorted_row_norms


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