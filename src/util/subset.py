"""
Contains a method for selecting d columns from a matrix
"""
import numpy as np

def select_d_random_columns(A, d, seed):
    """
    A - a matrix (nxm)
    d - the number of columns to get
    seed - for predictable randomness

    RETURN: B - where B is a (nxd) subset of A

    Given a matrix A, get x of its columns randomly
    """

    A_cols = A.shape[1]

    rng = np.random.default_rng(seed=seed)
    cols = rng.choice(A_cols, size=d, replace=False)

    sorted_cols = np.sort(cols)

    # converting to Compressed Sparse Column format for list slicing
    A_csc = A.copy()
    A_csc = A.tocsc()

    # Array slicing
    B = A_csc[:, sorted_cols]
    
    # TODO: should this be converted back to coo? 

    return B