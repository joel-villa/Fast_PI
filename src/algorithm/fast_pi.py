""" An implementation of an algorithm which is supposed to be faster than 
typical Power Iteration"""

import time
import scipy
import numpy as np
from collections.abc import Callable

from ..bounds.preprocess import preprocess
from ..bounds.util.comp_data import get_lambda_v
from ..util.sparse_rows import calc_row_norms

def plain_pi(
        A: scipy.sparse,
) -> tuple[float, np.ndarray]:
    """Do default power iteration

    Args:
        A (scipy.sparse): The matrix to get the top eigenvector/value of

    Returns:
        tuple[float, np.ndarray]: 
        float: top eigenvalue
        np.ndarray: top eigenvector
    """
    return get_lambda_v(A)
    

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
    diag_sampler = get_next_diagonal_sampler(
        row_norms=row_norms,
        rng=rng
    )

def fast_pi(
        A: scipy.sparse,
        num_samples: int,
) -> tuple[float, np.ndarray]:
    """Do the fast power-iteration method on the matrix A

    Args:
        A (scipy.sparse): The matrix to find the top spectral info about
        num_samples (int): How many iid copies of ~A

    Returns:
        tuple[float, np.ndarray]: 
        float: the top eigenvalue approximation
        np.ndarray: the top eigenvector approximation 
    """
    pass

def default_f(n: int) -> int:
    """The default sampling scheme

    Args:
        n (int): number of rows

    Returns:
        int: number of iid copies of ~A
    """
    pass

def bounded_fast_pi(
        A: scipy.sparse,
        epsilon: float,
        f_of_n:  Callable[[int], int] | None = None,
) -> tuple[float,np.ndarray, int, float]:
    """Do the fast power-iteration method on the matrix A with some allowable 
    ammount of error

    Args:
        A (scipy.sparse): The matrix to find the top spectral info about
        epsilon (float): Amount of allowable error in top eigenvalue
        f_of_n (Callable[[int], int] | None): A function that determines the number of 
        samples based on the matrix size
    Returns:
        tuple[float,np.ndarray, int, float]: 
        float: the top eigenvalue approximation
        np.ndarray: the top eigenvector approximation
        int: number of samples used
        float: probability of success (fallin within those epsilon bounds) 
    """
    pass

def rel_error(
        approx: float,
        true: float,
) -> float:
    """Get the relative error between the two values

    Args:
        approx (float): The approximate value
        true (float): The true value

    Returns:
        float: The relative error
    """
    return abs(approx - true) / abs(true)

if __name__ == '__main__':
    """Main for testing purposes
    """
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
    for mat in mats:
        print(mat)

        A, _ = preprocess(mat)

        pi_start = time.perf_counter()
        lambda_max, v_max = plain_pi(A)
        pi_end = time.perf_counter()
        pi_time = pi_end - pi_start

        epsilons = [0.25, 0.1, 0.05, 0.01]

        for epsilon in epsilons:
            fast_pi_start = time.perf_counter()
            lambda_guess, v_approx, num_samples, prob_success = bounded_fast_pi(
                A=A,
                epsilon=epsilon,
                f_of_n=default_f,
            )
            fast_pi_end = time.perf_counter()
            fast_pi_time = fast_pi_end - fast_pi_start


            lambda_approx = np.linalg.norm(A @ v_approx)

            rel_err = rel_error(
                approx=float(lambda_approx), 
                true=lambda_max,
            )

            print(f"With probability at least {prob_success}, using" 
                  f"{num_samples} iid copies of ~A, the top eigenvalue has"
                  fr"$||A\tilde w|| \in [(1 - {epsilon}) ||Aw||, ||Aw||]$")
            print(fr"||A \tilde w||  = {lambda_approx}"
                  fr"||A w||  = {lambda_max}")
            print(f"Relative error: {rel_err}")
            print(f"Time for plain PI: {pi_time}")
            print(f"Time for fast PI: {fast_pi_time}")