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
    diag_array = get_next_diagonal_sampler(
        row_norms=row_norms,
        rng=rng,
    )
    return diags(diag_array)

def round_row_norms(
        row_norms:np.ndarray,
) -> np.ndarray:
    """Check that the row_norms are not errorneous, if they have slight 
    rounding error, round them to one or zero where appropriate

    Args:
        row_norms (np.ndarray): Array of the row magnitudes of some scaled 
        matrix, s.t. its maximum row magnitude is one

    Returns:
        np.ndarray: The row_norms which should be used
    """
    # Ensure row norms are valid
    assert np.all(row_norms <= 1 + THIRTY_TWO_BIT_PRECISION), f"max row norm: {np.max(row_norms)}"
    assert np.all(row_norms >= 0 - THIRTY_TWO_BIT_PRECISION), f"min row norm: {np.min(row_norms)}"

    idx_too_large = [idx for idx, val in enumerate(row_norms) if val > 1]
    idx_too_small = [idx for idx, val in enumerate(row_norms) if val < 0]

    row_norms[idx_too_large] = 1
    row_norms[idx_too_small] = 0
    
    return row_norms

def get_row_norms(
        A:scipy.sparse.sparray,
) -> np.ndarray:
    """Get the row_norms of the given matrix

    Args:
        A (scipy.sparse.sparray): The matrix

    Returns:
        np.ndarray: The row magnitudes of every row in Af
    """
    row_norms = calc_row_norms(
            A=A,
            ord=2,
        )
    row_norms = round_row_norms(row_norms=row_norms)
    return row_norms
    
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
    row_norms = get_row_norms(A)
    diag_mat = get_diag_mat(
        row_norms=row_norms,
        rng=rng,
    )

    return diag_mat @ A

def init_A_tilde_sq(
       A:scipy.sparse.sparray, 
) -> tuple[scipy.sparse.csr_array, np.ndarray]:
    """Get the intial info necessary for generating ~A^TA

    Args:
        A (scipy.sparse.sparray): The matrix to approximate

    Returns:
        tuple[scipy.sparse.csr_array, np.ndarray]: 
        scipy.sparse.csr_array: a zeros array
        np.ndarray: the row_norms of A
    """
    zeros = scipy.sparse.csr_array(A.shape) # zeros sparse array
    row_norms = get_row_norms(A)

    return zeros, row_norms

def naive_A_tilde_sq(
        A:scipy.sparse.sparray,
        num_trials:int,
        rng:np.random.Generator,
) -> scipy.sparse.sparray:
    """Generate ~A^TA in a naive (non-paralellizable) way
    NOTE: ~A^TA = 1/N sum_{i=1}^N A^T D_i A
    where D_i is the i'th diagonal sampler

    Args:
        A (scipy.sparse.sparray): Some matrix to approximate A^TA of
        num_trials (int): The number of approximations of A^TA to generate
        rng (np.random.Generator): For repeatable randomization

    Returns:
        scipy.sparse.sparray: The approximation of A^TA
    """
    approximation, row_norms = init_A_tilde_sq(A)

    for i in range(num_trials):
        diag_i = get_diag_mat(
            row_norms=row_norms,
            rng=rng, #Same rng -> different independent diag_i
        )
        approximation += (1 / num_trials) * A.transpose() @ diag_i @ diag_i @ A

    return approximation

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
    approximation, row_norms = init_A_tilde_sq(A)

    for i in range(A.shape[0]):
        """Notice that everything that happens in this for loop is independent, 
        i.e. parallizable"""
        row_i = A[i, :] #i'th row of A
        norm_i = row_norms[i] #magnitude of i'th row 
        binom = np.random.binomial(
            n=num_trials,
            p=norm_i, # w.p. ||a_j||, X_ij is one
        )

        scalar = binom / (num_trials * norm_i)
        approximation += scalar * (row_i.transpose() @ row_i)
    return approximation

if __name__ == '__main__':
    """Yeahhh
    """
    from ...bounds.preprocess import preprocess
    from scipy.sparse.linalg import norm

    mats = [
        "1138_bus",
        "494_bus",
        "Harvard500",
        "bcspwr06",
        "bcsstk07",
        "bcsstk08",
        "bcsstk19",
        "bcsstk34",
        "bcsstm07",
        "blckhole",
        "cage7",
        "can_229",
        "dwt_193",
        "eris1176",
        "ex2",
        "fs_541_1",
        "gre_1107",
        "gre_343",
        "hor_131",
        "lshp1561",
        "msc00726",
        "nasa1824",
        "nos3",
        "tomography",
    ]

    mats = sorted(mats) #Alphabetical order
    print("Starting approximation tests...")
    N=8
    num_avg = 16
    print(f"N = {N}")
    print(f"num_avg = {num_avg}")
    for mat in mats:
        print(mat)
        A, _ = preprocess(mat_name=mat)
        rng = np.random.default_rng(seed=5334)
        rand_vect = rng.normal(loc=0.0, scale=0.0625, size=A.shape[0])
        fst_op_norm_avg = 0
        snd_op_norm_avg = 0
        for i in range(num_avg):
            fst_A_tilde_sq = naive_A_tilde_sq(
                A=A,
                num_trials=N,
                rng=rng,
            )
            fst_op_norm_avg += norm(fst_A_tilde_sq)
            snd_A_tilde_sq = get_A_tilde_sq(
                A=A,
                num_trials=N,
                rng=rng,
            )
            snd_op_norm_avg += norm(snd_A_tilde_sq)
        fst_op_norm_avg = fst_op_norm_avg / num_avg
        snd_op_norm_avg = snd_op_norm_avg / num_avg
            

        # Operator norm clarity check
        print(f"ACTUAL operator norm: {norm(A.transpose() @ A)}")
        print(f"naive operator norm: {fst_op_norm_avg}")
        print(f"distributable operator norm: {snd_op_norm_avg}\n")