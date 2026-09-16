"""For randomly generating those A approximations, based on their 
row-magnitudes"""

import scipy
from scipy.sparse import diags
import numpy as np

from ...util.sparse_rows import calc_row_norms
from ...util.constants import THIRTY_TWO_BIT_PRECISION


def get_next_diagonal_sampler(
        row_norms: np.ndarray,
        rng: np.random.Generator,
) -> np.ndarray:
    """Get the vector that represents a diagonal matrix which randomly 
    samples those columns of A

    Args:
        row_norms (np.ndarray): The norms of the rows of the matrix to sample
        rng (np.random.Generator): A random number generator

    Returns:
        np.ndarray: The vector which represents a random diagonal sampling 
        matrix
    """

    # Ensure row norms are valid
    assert np.all(row_norms <= 1 + THIRTY_TWO_BIT_PRECISION), f"max row norm: {np.max(row_norms)}"
    assert np.all(row_norms >= 0 - THIRTY_TWO_BIT_PRECISION), f"min row norm: {np.min(row_norms)}"

    idx_too_large = [idx for idx, val in enumerate(row_norms) if val > 1]
    idx_too_small = [idx for idx, val in enumerate(row_norms) if val < 0]

    row_norms[idx_too_large] = 1
    row_norms[idx_too_small] = 0

    binomial_vals = rng.binomial(n=1, p=row_norms)
    scaled_vals = binomial_vals / np.sqrt(row_norms)

    # print(f"row_norms: {row_norms[0:10]}")
    # print(f"binomial_vals: {binomial_vals[0:10]}")
    # print(f"scaled_vals: {scaled_vals[0:10]}")
    assert np.all((scaled_vals >= 1) | (scaled_vals == 0)) #TODO: can be deleted after testing once
    return scaled_vals


def get_diag_mat(
        row_norms: np.ndarray,
        rng: np.random.Generator,
) -> scipy.sparse.sparray:
    """Get an independent copy of the random diagonal matrix which serves the 
    purpose of matrix sampling

    Args:
        row_norms (np.ndarray): The norms of the rows of the matrix to sample
        rng (np.random.Generator): A random number generator

    Returns:
        scipy.sparse.sparray: The random diagonal sampling matrix
    """
    pass

def get_next_A_tilde(
        A: scipy.sparse.sparray,
        rng: np.random.Generator,
) -> scipy.sparse.sparray:
    """Generate the next reduced version of A

    Args:
        A (scipy.sparse.sparray): The original matrix
        rng (np.random.Generator): the random number generator

    Returns:
        scipy.sparse.sparray: The row-reduced version of A
    """
    row_norms = calc_row_norms(
        A=A,
        ord=2,
    )
    diag_vect = get_next_diagonal_sampler(
        row_norms=np.asarray(row_norms),
        rng=rng
    )
    diag_mat = diags(diag_vect)

    return diag_mat @ A

def naive_A_tilde_sq(
        A:scipy.sparse.sparray,
        num_trials:int,
        rng:np.random.Generator,
) -> scipy.sparse.sparray:
    """Generate ~A^TA in a naive (non-paralellizable) way

    Args:
        A (scipy.sparse.sparray): Some matrix to approximate A^TA of
        num_trials (int): The number of approximations of A^TA to generate
        rng (np.random.Generator): For repeatable randomization

    Returns:
        scipy.sparse.sparray: The approximation of A^TA
    """
    

def get_A_tilde_sq(
        A:scipy.sparse.sparray,
        num_trials:int,
        rng:np.random.Generator,
) -> scipy.sparse.sparray:
    """Generate ~A^TA in a (hopefully) more parallelizable way

    Args:
        A (scipy.sparse.sparray): Some matrix to approximate A^TA of
        num_trials (int): The number of approximations of A^TA to generate
        rng (np.random.Generator): For repeatable randomization

    Returns:
        scipy.sparse.sparray: The approximation of A^TA
    """
    pass