""" An implementation of an algorithm which is supposed to be faster than 
typical Power Iteration"""

import time
import scipy
import numpy as np
from collections.abc import Callable

from ..bounds.preprocess import preprocess
from ..bounds.util.comp_data import get_lambda_v
from ..bounds.mat_bernstein import get_mat_delta, get_mat_epsilon
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

def fast_pi(
        A: scipy.sparse,
        num_samples: int,
        seed: int,
) -> tuple[float, np.ndarray]:
    """Do the fast power-iteration method on the matrix A

    Args:
        A (scipy.sparse): The matrix to find the top spectral info about
        num_samples (int): How many iid copies of ~A
        seed (int): Seed of random number genrator

    Returns:
        tuple[float, np.ndarray]: 
        float: the top eigenvalue approximation
        np.ndarray: the top eigenvector approximation 
    """
    rng = np.random.default_rng(seed=seed)

    A_tildes = [get_next_A_tilde(A, rng) for _ in range(num_samples)]

    lambda_guesses = []
    v_guesses = []
    for A_tilde in A_tildes:
        lambda_guess, v_guess = get_lambda_v(A_tilde)
        lambda_guesses.append(lambda_guess)
        v_guesses.append(v_guess)

    lambda_guess = float(np.mean(lambda_guesses))
    v_approx = np.mean(v_guesses, axis=0)

    assert (v_approx.shape[0] == A.shape[0])

    return lambda_guess, v_approx

def default_f(n: int) -> int:
    """The default sampling scheme

    Args:
        n (int): number of rows

    Returns:
        int: number of iid copies of ~A
    """
    return int(np.ceil(np.log(n)**2))

def bounded_fast_pi(
        mat_name: str,
        A: scipy.sparse,
        epsilon: float,
        seed: int,
        f_of_n:  Callable[[int], int] | None = None,
) -> tuple[float,np.ndarray, int, float]:
    """Do the fast power-iteration method on the matrix A with some allowable 
    ammount of error

    Args:
        mat_name (str): The name of the matrix in the Suite Sparse collection
        A (scipy.sparse): The matrix to find the top spectral info about
        epsilon (float): Amount of allowable error in top eigenvalue
        f_of_n (Callable[[int], int] | None): A function that determines the number of 
        samples based on the matrix size
        seed (int): The seed for repeatable randomization
    Returns:
        tuple[float,np.ndarray, int, float]: 
        float: the top eigenvalue approximation
        np.ndarray: the top eigenvector approximation
        int: number of samples used
        float: probability of success (falling within those epsilon bounds) 
    """
    if f_of_n is None:
        f_of_n = default_f

    assert A.shape[0] == A.shape[1]

    num_approximations = f_of_n(A.shape[0])

    delta = get_mat_delta(
        mat_name=mat_name,
        N=num_approximations,
        epsilon=epsilon,
    )

    lambda_guess, v_approx = fast_pi(
        A=A,
        num_samples=num_approximations,
        seed=seed,
    )

    return (
        lambda_guess, 
        v_approx, 
        num_approximations,
        delta,
    )


def probable_fast_pi(
        mat_name: str,
        A: scipy.sparse,
        delta: float,
        seed: int,
        f_of_n:  Callable[[int], int] | None = None,
) -> tuple[float,np.ndarray, int, float]:
    """Do the fast power-iteration method on the matrix A with some allowable 
    ammount of error

    Args:
        mat_name (str): The name of the matrix in the Suite Sparse collection
        A (scipy.sparse): The matrix to find the top spectral info about
        delta (float): Probability of success
        f_of_n (Callable[[int], int] | None): A function that determines the number of 
        samples based on the matrix size
        seed (int): The seed for repeatable randomization
    Returns:
        tuple[float,np.ndarray, int, float]: 
        float: the top eigenvalue approximation
        np.ndarray: the top eigenvector approximation
        int: number of samples used
        float: With probability at least delta, the top eigenvector falls within 
        this epsilon ammount
    """
    if f_of_n is None:
        f_of_n = default_f

    assert A.shape[0] == A.shape[1]

    num_approximations = f_of_n(A.shape[0])

    epsilon = get_mat_epsilon(
        mat_name=mat_name,
        N=num_approximations,
        delta=delta,
    )

    lambda_guess, v_approx = fast_pi(
        A=A,
        num_samples=num_approximations,
        seed=seed,
    )

    return (
        lambda_guess, 
        v_approx, 
        num_approximations,
        epsilon,
    )
    

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
        "bcsstm12",
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
        deltas = [0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875]

        f_of_n = lambda n: int(np.ceil(np.sqrt(n)))

        for delta in deltas:
            try:
                fast_pi_start = time.perf_counter()
                lambda_guess, v_approx, num_samples, epsilon = probable_fast_pi(
                    mat_name=mat,
                    A=A,
                    delta=delta,
                    f_of_n=f_of_n,
                    seed=5334
                )
                fast_pi_end = time.perf_counter()
                fast_pi_time = fast_pi_end - fast_pi_start
    
    
                lambda_approx = np.linalg.norm(A @ v_approx)
    
                rel_err = rel_error(
                    approx=float(lambda_approx), 
                    true=lambda_max,
                )
    
                print(f"With probability at least {delta}, using" 
                      f"{num_samples} iid copies of ~A, the top eigenvalue has"
                      fr"$||A\tilde w|| \in [(1 - {epsilon}) ||Aw||, ||Aw||]$")
                print(fr"||A \tilde w||  = {lambda_approx}"
                      fr"||A w||  = {lambda_max}")
                print(f"Relative error: {rel_err}")
                print(f"Time for plain PI: {pi_time}")
                print(f"Time for fast PI: {fast_pi_time}")
            except Exception as e:
                print(f"Skipping delta={delta}; {e}")