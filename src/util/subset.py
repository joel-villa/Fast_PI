"""
Contains a method for selecting d columns from a matrix
"""
import numpy as np

from scipy.linalg import norm # 2-norm by default

from .scikit_jl import percent_reduce

def get_subset(A, cols):
    """Given a matrix A, get the columns specified by cols
    
    Args:
        A: a matrix (mxn)
        cols: a list of column indices to get (assumed sorted)

    RETURN: an (nxd) column subset of A, where d is the number of cols
    """
    # sorted_cols = np.sort(cols)

    # converting to Compressed Sparse Column format for list slicing
    A_csc = A.copy()
    A_csc = A.tocsc()

    # Array slicing
    B = A_csc[:, cols]
    
    # TODO: should this be converted back to coo? 

    return B

def weighted_select(n, d, seed, weights):
    """ Get d column indices, with weight based on weights

    Args:
        n: number of columns to select from
        d: number of columns to select
        seed: for predictable randomness
        weights: the weights to select columns with (should sum to 1)

    Return: a list of column indices, of length d (sorted)
    """
    rng = np.random.default_rng(seed=seed)
    cols = rng.choice(n, size=d, replace=False, p=weights)

    sorted_cols = np.sort(cols)

    return sorted_cols

def select_d_random_columns(A, d, seed):
    """Given a matrix A, get d of its columns randomly

    Args: 
        A: a matrix (mxn)
        d: the number of columns to get
        seed: for predictable randomness

    RETURN: B - an (nxd) column subset of A
    """

    num_cols = A.shape[1]

    # rng = np.random.default_rng(seed=seed)
    # cols = rng.choice(A_cols, size=d, replace=False)

    cols = weighted_select(
        n=num_cols, 
        d=d, 
        seed=seed, 
        weights=np.ones(num_cols)/num_cols
    )

    return get_subset(A=A, cols=cols)

def one_norm_select(A, d, seed):
    """Given a matrix A, get d of its columns randomly (putting more weight on 
    columns with higher 1-norm)

    Args: 
        A: a matrix (mxn)
        d: the number of columns to get
        seed: for predictable randomness

    RETURN: An (nxd) column subset of A
    """

    weights = np.asarray(np.sum(np.abs(A), axis=0)).ravel()# 1-norm of each column
    weights = weights / np.sum(weights) # normalize to get probabilities

    cols = weighted_select(
        n=A.shape[1], 
        d=d, 
        seed=seed, 
        weights=weights
    )


    # rng = np.random.default_rng(seed=seed)
    # cols = rng.choice(A_cols, size=d, replace=False, p=weights/np.sum(weights))

    return get_subset(A=A, cols=cols)

def two_norm_select(A, d, seed):
    """Given a matrix A, get d of its columns randomly (putting more weight on 
    columns with higher 2-norm)

    Based on Theorem 1.1 in 'Matrix Approximation and Projective Clustering via 
    Volume Sampling' by Amit Deshpande, Luis Rademacher, Santosh Vempala, 
    Grant Wang

    Args: 
        A: a matrix (mxn)
        d: the number of columns to get
        seed: for predictable randomness

    RETURN: An (nxd) column subset of A
    """

    weights = np.linalg.norm(A, ord=2, axis=0) # 2-norm of each column
    weights = weights * weights # square the 2-norm of the columns
    weights = weights / np.sum(weights) # normalize to get probabilities

    cols = weighted_select(
        n=A.shape[1], 
        d=d, 
        seed=seed, 
        weights=weights
    )

    return get_subset(A=A, cols=cols)

def nystrom_select(A, d, seed, gamma):
    """ Select those columns with a high gamma-ridge leverage score, as defined 
    in Definition 1. in https://arxiv.org/pdf/2604.20077
    
    Args: 
        A: a matrix (mxn)
        d: the number of columns to get
        gamma: the gamma parameter for the gamma-ridge leverage score
        seed: for predictable randomness
        
    RETURN: An (nxd) column subset of A
    """
    B = A + gamma * np.eye(A.shape[0])
    #TODO: presumably inefficient (not being accounted for in scalar mult count)
    B_inv = np.linalg.inv(B) 
    
    #i'th column of original matrix dot product with i'th column of B_inv
    # NOTE: '*' in numpy is elementwise multiplication
    dot_prod_matrix = A * B_inv
    leverage_scores = np.asarray(np.sum(dot_prod_matrix, axis=0)).ravel()

    effective_dimension = np.sum(leverage_scores)

    if d < effective_dimension:
        print(f"WARNING: d = {d} < {effective_dimension} = effective dimension; may not be selecting enough columns for good approximation")

    weights = leverage_scores / effective_dimension

    cols = weighted_select(
        n=A.shape[1], 
        d=d, 
        seed=seed, 
        weights=weights
    )

    return get_subset(A=A, cols=cols)

def cost_nystrom(A, d, seed, gamma):
    """ Get the cost of doing nystrom column selection, in terms of scalar 
    multiplications

    Args: 
        A: a matrix (mxn)
        d: the number of columns to get
        gamma: the gamma parameter for the gamma-ridge leverage score
        seed: for predictable randomness
        
    Return: the cost of doing nystrom column selection, in terms of scalar 
            multiplications
    """
    n, m = A.shape
    
    # Cost of computing B_inv (not accounting for sparsity)
    cost_B_inv = (n**3) 

    # Cost of computing dot_prod_matrix (not accounting for sparsity)
    cost_dot_prod = n * m

    return cost_B_inv + cost_dot_prod


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
