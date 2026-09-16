"""Contains the newest sets of tests (has provable closeness results)"""

import scipy
import numpy as np

from .util import comp as comp
from .util.approx import get_next_A_tilde, naive_A_tilde_sq, get_A_tilde_sq
from .util.work import power_work, work_naive_pessimistic, work_fast
from ..util.power import power, rayleigh_quotient


def baseline(
        A: scipy.sparse.sparray,
        v0: np.ndarray,
        max_iter: int,
        tol: float,
) -> tuple [np.ndarray, np.ndarray, str]:
    """Run the default power iteration on this matrix

    Args:
        A: the original matrix
        v0: initial guess for top eigenvector
        max_iter: maximum number of iterations to do power iteration
        tol (float): how much precision before terminating power

    Returns:
        tuple [np.ndarray, np.ndarray, str]: 
        np.ndarray: the x-values (the ammount of work done),
        np.ndarray: the y-values (the score of the vector),
        str: the string representation of this test
    """
    v=v0
    scores, _, iter = comp.init_test(
        A=A,
        v0=v0,
        max_iter=max_iter,
    )
    while iter < max_iter:
        lam, v = power(
            A = A,
            v0=v,
            num_iter=1,
        )
        scores[iter] = lam
        if (comp.rel_error(scores[iter], scores[iter - 1]) < tol):
            iter += 1
            break
        iter += 1

    scores = scores[0:iter]
    work = power_work(matrix=A, num_iter=iter)
    lbl = "baseline"
    return work, scores, lbl

def naive_test(
        A: scipy.sparse.sparray,
        v0: np.ndarray,
        max_iter: int,
        tol: float,
        num_trials:int,
        seed:int,
        is_distributed:bool
) -> tuple [np.ndarray, np.ndarray, str]:
    """Run the less-parallelizable approach, return results

    Args:
        A (scipy.sparse.sparray): Matrix in question
        v0 (np.ndarray): Initial guess for top eigenvector
        max_iter (int): Max number of iterations for power
        tol (float): Tolerance of power iteration
        num_trials (int): Number of things to average
        seed (int): For repeatable randomization
        is_distributed (bool): Changes ammount of work that goes into computing 
        an approximation for A^TA
        
    Returns:
        tuple [np.ndarray, np.ndarray, str]: 
        np.ndarray: work per iteration
        np.ndarray: score of approximate eigenvector per iteration
        str: label of this test
    """
    rng = np.random.default_rng(seed=seed)
    A_sq = A.transpose() @ A
    tilde_A_sq = naive_A_tilde_sq(
        A=A, 
        num_trials=num_trials,
        rng=rng,
    )
    init_work = work_naive_pessimistic(
        A=A,
        num_tirals=num_trials,
        is_distributed=is_distributed
    )
    scores, _, i = comp.init_test(
        A=A,
        v0=v0,
        max_iter=max_iter,
    )
    v=v0

    while i < max_iter:
        _, v = power(
            A=tilde_A_sq,
            v0=v,
            num_iter=1,
        )
        scores[i] = rayleigh_quotient( # Score of approximation based on actual
            x=v,
            A=A_sq,
        )
        if (comp.rel_error(scores[i], scores[i - 1]) < tol): #TODO: this was copied and pasted, bad coding practice, maybe add a helper function for rel error check
            i += 1
            break
        i += 1

    scores = scores[0:i]
    work = power_work(matrix=A, num_iter=i)
    work += init_work
    lbl = f"naive (distributed={is_distributed}), {num_trials} trials" 
    return work, scores, lbl #TODO: untested


def distributable_test(
        A: scipy.sparse.sparray,
        v0: np.ndarray,
        max_iter: int,
        tol: float,
        num_trials:int,
        seed:int,
        is_distributed:bool
) -> tuple [np.ndarray, np.ndarray, str]:
    """Run the parallelizable approach, return results

    Args:
        A (scipy.sparse.sparray): Matrix in question
        v0 (np.ndarray): Initial guess for top eigenvector
        max_iter (int): Max number of iterations for power
        tol (float): Tolerance of power iteration
        num_trials (int): Number of things to average
        seed (int): For repeatable randomization
        is_distributed (bool): Changes ammount of work that goes into computing 
        an approximation for A^TA

    Returns:
        tuple [np.ndarray, np.ndarray, str]: 
        np.ndarray: work per iteration
        np.ndarray: score of approximate eigenvector per iteration
        str: label of this test
    """
    #TODO: implement this + main
    pass