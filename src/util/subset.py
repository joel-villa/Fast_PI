"""
Contains a method for selecting d columns from a matrix
"""
import numpy as np

from .scikit_jl import percent_reduce

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

def get_reduce_funct(type):
    """ Get the reduction function based on the type of reduction
    
    Args:
        type: the type of reduction to do (string representation)

    Return: the reduction function"""
    match type.lower():
        case "simple":
            # simple random sampling of columns
            return select_d_random_columns
        case "1-norm":
            # random sampling of columns, with weight based on 1-norm of column
            return  one_norm_select
        case _:
            raise TypeError(f"Invalid sampling type: {type}")
        
def reduce_A(A, p, seed, type):
    """ Reduce A based on percent reduction and reduction function
    
    Args:
        A: the matrix to reduce
        p: reduction percentage
        seed: for repeatable randomness
        type: the type of reduction to do (string representation)
    
    Return: the reduced A
    """
    _, n = A.shape

    d = percent_reduce(n=n, p=p)
    reduce = get_reduce_funct(type)
    return reduce(A, d, seed)
