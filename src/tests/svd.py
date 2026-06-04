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

from ..util.subset import p_reduce_A
from ..util.sparsify import p_sparse

def svd_test(A, seed, max_k, err_type, A_tilde=None):
    """ Baseline test of svd on the matrix A

    Args:
        A: the original matrix (in CSR format)
        seed: for repeatable randomness
        max_k: the maximum k for top k singular values to approximate
        err_type: '2' -> 2-norm error 
                  'fro' -> Frobenius norm
        A_tilde: the reduced version of A

    Return:
        xs: k in range (1, 2, ...) where k is the number of top k singular values
            being approximated
        ys: an array of error of one of the two types:
            (1) ||A - A_k||_F / ||A||_F 
                this is akin to treating mat as vector and taking 2-norm
            (2) ||A - A_k||_2 / ||A||_2  
                diff in top singular value 
        lbl: the string representation of this test

    TODO: doesn't work w/ dimensionality reductions, is there a way to get 
    arround this? Maybe pad with zeroes? But that way would still not work for 
    jl reductions

    POTENTIAL (UNVERIFIED) SOLUTION: 
        A_k = (U U^T) A
        Where U is the one found by svd on A_tilde

    TODO: big bug, always return 1, other than last k-value, what's up with that? 
    """
    if A_tilde is None:
        # BASELINE TEST
        A_tilde = A
    # TODO: does spla.svds() work w/ dense matrices? 
    # if type(A) is not np.ndarray:
    #     # A is in sparse format, use spla.svds
    #     U, S, Vt = spla.svds(A, k=max_k, rng=seed)
    # else:
    #     # A is in dense format, use np.linalg.svd
    #     U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Get the top k singular values and vectors of A_tilde
    U, S, Vt = spla.svds(A=A_tilde, k=max_k, rng=seed)

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


def sparse(A, seed, max_k, err_type, spa_type, p):
    """ Sparsification test

    Args:
        A: the original matrix (in CSR format)
        seed: for repeatable randomness
        max_k: the maximum k for top k singular values to approximate
        err_type: '2' -> 2-norm error 
                  'fro' -> Frobenius norm
        spa_type: 'mb' -> magnitude based sparsification (variable per entry)
                  'sd' -> scale diagonal sparsification
                  'cd' -> constant diagonal sparsification
                  'generic' -> default behavior
        p: percent sparsified (in expectation)

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
    A_reduced = p_sparse(
        A=A,
        p=p,
        type=spa_type,
        seed=seed,
    )


    xs, ys, lbl = svd_test(
        A=A,
        seed=seed,
        max_k=max_k,
        err_type=err_type,
        A_tilde=A_reduced,
    )
    
    new_zeros = A.nnz - A_reduced.nnz

    return xs, ys, f"{p}% {spa_type} sparsified {A.shape} ({new_zeros} new zeros)"

def subset(A, seed, max_k, err_type, sub_type, p, gamma):
    """ Test behavior of subset reduction on svd
    
    Args: 
        A: the original matrix (in CSR format)
        seed: for repeatable randomness
        max_k: the maximum k for top k singular values to approximate
        err_type: '2' -> 2-norm error 
                  'fro' -> Frobenius norm
        sub_type: type of subset selection
        p: percent reduction
        gamma: for Nystrom gamma-leverage-score sampling

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
    A_reduced = p_reduce_A(
        A=A, 
        p=p, 
        seed=seed, 
        type=sub_type, 
        gamma=gamma,
    )

    xs, ys, lbl = svd_test(
        A=A, 
        seed=seed, 
        max_k=max_k, 
        err_type=err_type, 
        A_tilde=A_reduced,
    )

    return xs, ys, f"{100 - p}% {lbl}"


