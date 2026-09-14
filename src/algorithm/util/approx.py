"""For randomly generating those A approximations, based on their 
row-magnitudes"""

import scipy
import numpy as np

from ..util.sparse_rows import calc_row_norms


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
    assert np.all(row_norms <= 1)
    assert np.all(row_norms >= 0)

    binomial_vals = rng.binomial(n=1, p=row_norms)
    scaled_vals = binomial_vals / np.sqrt(row_norms)

    # print(f"row_norms: {row_norms[0:10]}")
    # print(f"binomial_vals: {binomial_vals[0:10]}")
    # print(f"scaled_vals: {scaled_vals[0:10]}")
    assert np.all((scaled_vals >= 1) | (scaled_vals == 0)) #TODO: can be deleted after testing once
    return scaled_vals
    

def get_next_A_tilde(
        A: scipy.sparse,
        rng: np.random.Generator,
) -> scipy.sparse:
    """Generate the next reduced version of A

    Args:
        A (scipy.sparse): The original matrix
        rng (np.random.Generator): the random number generator

    Returns:
        scipy.sparse: The row-reduced version of A
    """
    row_norms = calc_row_norms(
        A=A,
        ord=2,
    )
    diag_vect = get_next_diagonal_sampler(
        row_norms=row_norms,
        rng=rng
    )
    diag_mat = np.diag(diag_vect)

    return diag_mat @ A