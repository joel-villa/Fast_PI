"""
A file for some tests, which return xs and ys (for easy plotting)
"""
from math import ceil 
from ..test_util import eig_functs as eig
from ..test_util import scikit_jl as jl
import numpy as np


def jl_top_eig_pres(A, ps, seed):
    """
    Test how well johnson lindenstrauss maintains top eigenvector of 
    sparse matrices for variable epsilon (using personal implementation)

    A  - sparse matrix in COO format
    ps - percent sparsified
    
    RETURN: ps - input
            ys - eigenvector preservation
    """

    ys = np.zeros(np.shape(ps))
    xs = np.zeros(np.shape(ps))

    for i, p in enumerate(ps):
        A_reduced = jl.percent_reduce(A, p, seed=seed)
        diff = eig.diff_in_top_eigs(A, A_reduced)
        ys[i] = diff
        xs[i] = A_reduced.shape[1]

    return ps, ys

def scikit_eig_percent_reduce(A, ps, seed, type="jl_gaussian"):
    """
    Test how well johnson lindenstrauss maintains top eigenvector of 
    sparse matrices for variable epsilon (using sklearn gaussian projection
    library)

    A  - sparse matrix in COO format
    ps - percent reduction ammounts 
    
    RETURN: xs - reduced dimension (or maybe epsilon-TBD)
            ys - eigenvector preservation
    """

    match type:
        case "jl_gaussian":
            reduct_funct = jl.jl_gaussian
        case "jl_sparse":
            reduct_funct = jl.jl_sparse
        case _:
            raise TypeError(f"type must be 'jl_gaussian' or 'jl_sparse', not {type}")

    ys = np.zeros(np.shape(ps))
    xs = np.zeros(np.shape(ps))

    for i, p in enumerate(ps):
        n = A.shape[1]
        reduce_ammount = ceil(n * p * 0.01)
        d = n - reduce_ammount
        if (d <= 1):
            print(f"WARNING: reduced dimension {d} is too small, defaulting to 2")
            d = 2
        
        A_reduced = reduct_funct(A, d=d, eps=0.9, seed=seed)

        diff = eig.diff_in_top_eigs(A, A_reduced)
        ys[i] = diff
        xs[i] = A_reduced.shape[1]
    
    return ps, ys

def scikit_jl_top_eig_pres(A, epsilons, seed):
    """
    Test how well johnson lindenstrauss maintains top eigenvector of 
    sparse matrices for variable epsilon (using sklearn gaussian projection
    library)

    A        - sparse matrix in COO format
    epsilons - eigenvalues 
    
    RETURN: xs - reduced dimension (or maybe epsilon-TBD)
            ys - eigenvector preservation
    """

    ys = np.zeros(np.shape(epsilons))
    xs = np.zeros(np.shape(epsilons))

    for i, eps in enumerate(epsilons):
        A_reduced = jl.jl_gaussian(A, d="auto", eps=eps, seed=seed)
        diff = eig.diff_in_top_eigs(A, A_reduced)
        ys[i] = diff
        xs[i] = A_reduced.shape[1]
    
    return xs, ys
    
    

    
