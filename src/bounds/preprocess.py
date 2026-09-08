"""Theoretical results are reliant on matrices being scaled s.t. the max row
magnitude is 1"""

import scipy

import numpy as np

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

    #Don't want complex matrices
    if (A is None):
        raise TypeError(f"{mat_name}not found")
    elif (A.data.dtype not in [np.float32, np.float64]):
        if (A.data.dtype not in [np.int32, np.int64]):
            raise TypeError(f"Unexpected Type: {A.data.dtype}")
        A.data = A.data.astype(float)

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{mat_name} non-square: A.shape = {A.shape}")
    
    max_row_magnitude = get_sorted_row_norms(mat_name, rescale=False)[0]

    # Rescale A
    A = A / max_row_magnitude

    A.eliminate_zeros()

    return A, max_row_magnitude