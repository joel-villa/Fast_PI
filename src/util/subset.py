"""
Contains a method for selecting d columns from a matrix
"""
import numpy as np

def get_subset(A, cols):
    """Given a matrix A, get the columns specified by cols
    
    Args:
        A: a matrix (nxm)
        cols: a list of column indices to get

    RETURN: an (nxd) column subset of A, where d is the number of cols
    """
    sorted_cols = np.sort(cols)

    # converting to Compressed Sparse Column format for list slicing
    A_csc = A.copy()
    A_csc = A.tocsc()

    # Array slicing
    B = A_csc[:, sorted_cols]
    
    # TODO: should this be converted back to coo? 

    return B

def select_d_random_columns(A, d, seed):
    """Given a matrix A, get d of its columns randomly

    Args: 
        A: a matrix (nxm)
        d: the number of columns to get
        seed: for predictable randomness

    RETURN: B - an (nxd) column subset of A
    """

    A_cols = A.shape[1]

    rng = np.random.default_rng(seed=seed)
    cols = rng.choice(A_cols, size=d, replace=False)

    return get_subset(A=A, cols=cols)

def one_norm_select(A, d, seed):
    """Given a matrix A, get d of its columns randomly (putting more weight on 
    columns with higher 1-norm)

    Args: 
        A: a matrix (nxm)
        d: the number of columns to get
        seed: for predictable randomness

    RETURN: An (nxd) column subset of A
    """

    A_cols = A.shape[1]
    weights = np.asarray(np.sum(np.abs(A), axis=0)).ravel()# 1-norm of each column

    rng = np.random.default_rng(seed=seed)
    cols = rng.choice(A_cols, size=d, replace=False, p=weights/np.sum(weights))

    return get_subset(A=A, cols=cols)