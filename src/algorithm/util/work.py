"""For generating ammount of work to compute an average of approximations of 
A^TA"""
import scipy
import numpy as np

def power_work(
        matrix: scipy.sparse.sparray,
        num_iter: int,
) -> np.ndarray:
    """The amount of cummulative work done at every iteration of Power on the 
    given matrix

    Args:
        matrix (scipy.sparse.sparray): The matrix that power-iteration was run on
        num_iter (int): number of iterations run

    Returns:
        np.ndarray: A linearly increasing one-dimensional array of length 
        num_iter
    """
    nnzs = matrix.nnz
    iterations = np.arange(0, num_iter)
    total_work = iterations * nnzs # Doing SpMv multiplication per iteration
    return total_work

def work_naive_pessimistic(
    A:scipy.sparse.sparray,
    num_tirals:int,
    is_distributed:bool,
) -> int:
    """Calculate ammount of work (scalar mults) involved in computing this 
    average in a naive way

    NOTE: actually computing this work involves expectation, and sparsity 
    rate, this is a simple overestimate

    Args:
        A (scipy.sparse.sparray): The matrix to approximate
        num_tirals (int): The ammount of approximations to average over
        is_distributed (bool): True -> taking maxium, False -> summing up work

    Returns:
        int: amount of scalar mults
    """
    # Doing A.transpose() @ diag_i @ diag_i @ A every iteration
    # mults_per_iter = A.nnz * 
    pass

def work_fast_pessimistic(
        A:scipy.sparse.sparray,
        num_trials:int,
        is_distributed:bool,
) -> int:
    """Calculate ammount of work required to do the parallelizable calculation 
    of an averaged approximation of A^TA

    NOTE: actually computing this work involves expectation, and sparsity 
    rate, this is a simple overestimate

    Args:
        A (scipy.sparse.sparray): The matrix in question
        num_trials (int): The ammount of trials to average over
        is_distributed (bool): True -> taking maxium, False -> summing up work 
        per row

    Returns:
        int: Ammount of work per thread
    """
    pass