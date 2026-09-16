"""Contains the newest sets of tests (has provable closeness results)"""

import scipy
import numpy as np

from .util.approx import get_next_A_tilde
from ..util.power import power
from .util import comp as comp


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
    for iter in range(1, max_iter):
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
    work = comp.power_work(matrix=A, num_iter=iter)
    lbl = "baseline"
    return work, scores, lbl

def naive_test(
        A: scipy.sparse.sparray,
        v0: np.ndarray,
        max_iter: int,
        tol: float,
        num_trials:int,
        seed:int,
) -> tuple [np.ndarray, np.ndarray, str]:
    """Run the non-parallelizable approach, return results

    Args:
        A (scipy.sparse.sparray): Matrix in question
        v0 (np.ndarray): Initial guess for top eigenvector
        max_iter (int): Max number of iterations for power
        tol (float): Tolerance of power iteration
        num_trials (int): Number of things to average
        seed (int): For repeatable randomization

    Returns:
        tuple [np.ndarray, np.ndarray, str]: 
        np.ndarray: work per iteration
        np.ndarray: score of approximate eigenvector per iteration
        str: label of this test
    """

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
    pass