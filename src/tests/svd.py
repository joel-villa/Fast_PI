"""
Testing how the different reduction techniques impact singular value 
decomposition.

All methods in this file return three things:
    xs: k in range (1, 2, ...) where k is the number of top k singular values
        being approximated
    ys: an array of error of one of the two types:
        (1) ||A - A_k||_F / ||A||_F 
            this is akin to treating mat as vector and taking 2-norm
        (2) ||A - A_k||_2 / ||A||_2  
            diff in top singular value 
    lbl: the string representation of this test
"""

import scipy.sparse.linalg as spla
import numpy as np

def svd_test(A, seed, max_k, err_type):
    """ Baseline test of svd on the matrix A

    Args:
        A: the matrix to do power iteration on (in CSR format)
        seed: for repeatable randomness
        max_k: the maximum k for top k singular values to approximate
        err_type: '2' -> 2-norm error 
                  'fro' -> Frobenius norm

    Return:
        xs: k in range (1, 2, ...) where k is the number of top k singular values
            being approximated
        ys: an array of error of one of the two types:
            (1) ||A - A_k||_F / ||A||_F 
                this is akin to treating mat as vector and taking 2-norm
            (2) ||A - A_k||_2 / ||A||_2  
                diff in top singular value 
        lbl: the string representation of this test
    """
    # TODO: does spla.svds() work w/ dense matrices? 
    # if type(A) is not np.ndarray:
    #     # A is in sparse format, use spla.svds
    #     U, S, Vt = spla.svds(A, k=max_k, rng=seed)
    # else:
    #     # A is in dense format, use np.linalg.svd
    #     U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Get the top k singular values and vectors of A
    U, S, Vt = spla.svds(A, k=max_k, rng=seed)

    # Compute the error for each k
    xs = []
    ys = []
    for k in range(1, max_k + 1):
        U_k = U[:, :k] # First k columns of U
        S_k = np.diag(S[:k]) # First k singular values (S is 1D)
        Vt_k = Vt[:k, :] # First k rows of Vt
        A_k = U_k @ S_k @ Vt_k # rank-k approximation of A

        diff_mat = A - A_k

        error = np.linalg.norm(diff_mat, ord=err_type) / spla.norm(A, ord=err_type)

        xs.append(k)
        ys.append(error)

    lbl = "Baseline SVD"

    return xs, ys, lbl

#TODO: sparse, jl, and subset tests