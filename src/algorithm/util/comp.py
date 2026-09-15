"""Compute those things of interest when testing the efficacy of Fast PI"""

import scipy
import numpy as np

from ...util.power import power, rayleigh_quotient
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

def init_test(
        A:scipy.sparse,
        v0:np.ndarray,
        max_iter:int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Initialize this power iteration test

    Args:
        A (scipy.sparse): The matrix in question
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

    scores[0] = rayleigh_quotient(x=v0, A=A_tilde)
    vects[0] = v0
    iter = 1

    return scores, vects, iter

def test_A_tilde(
        A_tilde: scipy.sparse,
        v0: np.ndarray,
        max_iter: int,
        tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Get the approximate spectral information at every iteration

    Args:
        A_tilde (scipy.sparse): The approximation matrix in question
        v0 (np.ndarray): Some original guess for the top eigenvector
        max_iter (int): The maximum number of iterations to run power iteration
        tol (float): The ammount of allowable error

    Returns:
        tuple[np.ndarray, np.ndarray]: 
        np.ndarray: the score of the top eigenvector approximation at every 
        iteration (note that this is a 1xnum_iter array, where num_iter is the 
        number of iterations until Power terminates)
        np.ndarray: the top eigenvector approximation at every iteration 
        (num_iterxn)
    """
    v=v0
    scores, vects, iter = init_test(
        A=A_tilde,
        v0=v0,
        max_iter=max_iter,
    )
    
    while iter < max_iter:
        lam, v = power(
            A = A_tilde,
            v0=v,
            num_iter=1,
        )
        scores[iter] = lam
        vects[iter] = v
        if (rel_error(scores[iter], scores[iter - 1]) < tol):
            iter += 1
            break
        iter += 1
        
    scores = scores[0:iter]
    vects = vects[0:iter]
    return scores, vects

def power_work(
        matrix: scipy.sparse,
        num_iter: int,
) -> np.ndarray:
    """The amount of cummulative work done at every iteration of Power on the 
    given matrix

    Args:
        matrix (scipy.sparse): The matrix that power-iteration was run on
        num_iter (int): number of iterations run

    Returns:
        np.ndarray: A linearly increasing one-dimensional array of length 
        num_iter
    """
    nnzs = A.nnz
    iterations = np.arange(0, num_iter)
    total_work = iterations * nnzs # Doing SpMv multiplication per iteration
    return total_work

if __name__ == '__main__':
    """Yeahhh
    """
    from ...bounds.preprocess import preprocess
    from .approx import get_next_A_tilde

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
        top_lambdas, top_vs = test_A_tilde(
            A_tilde=A_tilde,
            v0=rand_vect,
            max_iter=20,
            tol=1/128,
        )
        work = power_work(
            matrix=A,
            num_iter=top_lambdas.shape[0],
        )
        print(f"num_iter: {top_lambdas.shape[0]}")
        print(f"top_lambdas: {top_lambdas[-7:]}\ntop_vs: {top_vs[-7:]}\nwork: {work[-7:]}")