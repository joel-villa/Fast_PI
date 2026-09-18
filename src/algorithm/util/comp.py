"""Compute those things of interest when testing the efficacy of Fast PI"""

import scipy
import numpy as np

from ...util.power import power, rayleigh_quotient
from .work import power_work
from ...util.constants import THIRTY_TWO_BIT_PRECISION

def rel_score(
        max:float,
        xs:np.ndarray,
) -> np.ndarray:
    """Adjust the x values s.t. they are on a range from 0 to 1, where if x_i 
    equals max then set that value to one

    Args:
        max (float): The maximum possible value
        xs (np.ndarray): The list of values

    Returns:
        np.ndarray: Those adjust values 
    """
    # Error handlign
    assert np.max(xs) < max + THIRTY_TWO_BIT_PRECISION, f"np.max(xs) = {np.max(xs)} > {max}"
    assert np.all(xs >= 0), f"Scores must be nonnegative"

    return xs / max

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
    rel_error = abs(approx - true) / abs(true)
    # print(f"approx - true / approx = {abs(approx - true)} / {abs(true)} = {rel_error}")
    return rel_error

def online_avg(
        old_avg:np.ndarray,
        new_val:np.ndarray,
        new_total:int,
) -> np.ndarray:
    """Generate the new average given the old + the new + the new total count

    Args:
        old_avg (scipy.sparse): The current average
        new_val (scipy.sparse): The value which is updating the average
        new_total (int): The grand total ammount

    Returns:
        scipy.sparse: The new average
    """

    return old_avg + ((new_val - old_avg) / new_total)

def init_test(
        A:scipy.sparse.sparray,
        v0:np.ndarray,
        max_iter:int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Initialize this power iteration test

    Args:
        A (scipy.sparse.sparray): The matrix in question
        v0 (np.ndarray): The initial guess for the eigenvector
        max_iter (int): the maximum number of iterations

    Returns:
        tuple[np.ndarray, np.ndarray, int]: 
        np.ndarray: the array which will hold the top eigenvalues
        np.ndarray: the array which will hold the top eigenvectors
        int: the initial number of iterations
    """
    scores = np.zeros(shape=(max_iter,))
    vects = np.zeros(shape=(max_iter,v0.shape[0]))

    scores[0] = rayleigh_quotient(x=v0, A=A)
    vects[0] = v0
    iter = 1

    return scores, vects, iter



if __name__ == '__main__':
    """Yeahhh
    """
    from ...bounds.preprocess import preprocess
    from .approx import get_next_A_tilde
    from ..non_proven_tests import test_A_tilde

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
        A, _ = preprocess(mat_name=mat)
        rng = np.random.default_rng(seed=5334)
        rand_vect = rng.normal(loc=0.0, scale=0.0625, size=A.shape[0])
        A_tilde = get_next_A_tilde(A, rng=rng)
        top_lambdas, top_vs, iter = test_A_tilde(
            A_tilde=A_tilde,
            v0=rand_vect,
            max_iter=20,
            tol=1/128,
        )
        top_vs = top_vs[0:iter]
        work = power_work(
            matrix=A,
            num_iter=top_lambdas.shape[0],
        )
        print(f"num_iter: {top_lambdas.shape[0]}")
        print(f"top_lambdas: {top_lambdas[-7:]}\ntop_vs: {top_vs[-7:]}\nwork: {work[-7:]}")