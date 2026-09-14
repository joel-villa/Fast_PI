"""All functions here have the same return type (xs, ys, lbl)"""

#TODO: max work
import scipy
import numpy as np

def test_averaging(
        A:scipy.sparse,
        u_0:np.ndarray,
        s_star:float,
        max_iter:int,
        seed:int,
        tol:float,
        num_samples:int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Test an averaging approach, generate N, approximations of A, use them 
    to independently get N seperate approximations for v_star, average them, 
    hopefully getting better results

    Args:
        A: the original matrix
        u_0: initial guess for top left eigenvector
        s_star: actual top singular value
        max_iter: maximum number of iterations to do power iteration
        tol: how much tolerance (for stopping condition of power iteration)
        seed: for repeatable randomness (scikit does not have repeatable 
              randomness)
        tol (float): how much precision before terminating power
        num_samples (int): How many approximations of A?

    Returns:
        tuple[np.ndarray, np.ndarray, str]: 
        np.ndarray: the x-values (the ammount of work done),
        np.ndarray: the y-values (the score of the vector),
        str: the string representation of this test
    """
    rng = np.random.default_rng(seed=seed)

    # for i in range(num_samples):
        