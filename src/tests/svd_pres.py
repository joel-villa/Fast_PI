"""
A file containing tests for singular vector preservation 

Functions in this file have Return type as follows:
    xs: an array of reduction percentages (ps)
    ys: a 2d array of relative error in top k left singular vectors
    lbl: a string representation of this test
"""
import numpy as np

from ..util import svd_pres as util

from ..util.subset import p_reduce_A
from ..util.sparsify import p_sparse
from ..util.proven import get_A_tilde

def pi_inc_pow(A, v0, v_star, k, seed, pows, gen_type, sf_kwargs):
    """ Test of random col sampling based on percent of reduction

    Args:
        A: the matrix to do power iteration on
        v0: initial guess for top right singular vectors
        v_star: The top-k right singular vectors of the original matrix
        k: The number of singular vectors to compute
        seed: for repeated randomness
        ps: reduction percentages
        gen_type: string representation of reduction type
    Return:
        xs: an array of reduction percentages (ps)
        ys: a 2d array of relative error in top k left singular vectors 
            (each column represents a distinct k-value)
        lbl: a string representation of this test
    """
    ys_shape = (pows.shape[0], k)
    ys = np.zeros(ys_shape)

    for i, pow in enumerate(pows):
        pow_kwargs = sf_kwargs | {"power": pow}
        A_tilde = get_A_tilde(
            A=A,
            gen_type=gen_type,
            sf_kwargs=pow_kwargs,
            seed=seed
        )
        ys_i = util.eig_pres_right(
            A_tilde=A_tilde,
            v0=v0,
            v_star=v_star,
            k=k,
            seed=seed,
        )
        ys[i] = ys_i
    
    return pows, ys, gen_type

def subset_p(A, u_0, u_star, k, seed, ps, type, gamma):
    """ Test of random col sampling based on percent of reduction

    Args:
        A: the matrix to do power iteration on
        u_0: initial guess for top left singular vectors
        u_star: The top-k left singular vectors of the original matrix
        k: The number of singular vectors to compute
        seed: for repeated randomness
        ps: reduction percentages
        type: string representation of reduction type
        gamma: gamma parameter for the gamma-ridge leverage score
    Return:
        xs: an array of reduction percentages (ps)
        ys: a 2d array of relative error in top k left singular vectors 
            (each column represents a distinct k-value)
        lbl: a string representation of this test
    """
    ys_shape = (ps.shape[0], k)
    ys = np.zeros(ys_shape)

    for i, p in enumerate(ps):
        A_reduced = p_reduce_A(
            A=A, 
            p=p,
            seed=seed,
            type=type,
            gamma=gamma,
        )
        ys_i = util.eig_pres_left(
            A_tilde=A_reduced,
            u0=u_0,
            u_star=u_star,
            k=k,
            seed=seed,
        )
        ys[i] = ys_i
    
    return ps, ys, type

def sparse_p(A, u_0, u_star, k, seed, ps, type):
    """ Preservation of top k eigenvectors of a sparsified version of A

    Args: 
        A: the original matrix
        u_0: initial guess for top left singular vectors of A
        u_star: The top-k left singular vectors of the original matrix
        k: The number of singular vectors to compute
        seed: for repeated randomness
        ps: reduction percentages
        type: string representation of sparsification type

    Returns:
        xs: an array of reduction percentages (ps)
        ys: a 2d array of relative error in top k left singular vectors 
            (each column represents a distinct k-value)
        lbl: a string representation of this test
    """
    ys_shape = (ps.shape[0], k)
    ys = np.zeros(ys_shape)

    for i, p in enumerate(ps):
        A_sparse = p_sparse(
            A=A, 
            p=p,
            seed=seed,
            type=type,
        )
        ys_i = util.eig_pres_left(
            A_tilde=A_sparse,
            u0=u_0,
            u_star=u_star,
            k=k,
            seed=seed,
        )
        ys[i] = ys_i
    
    return ps, ys, type
