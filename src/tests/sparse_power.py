from Sparsification_Research.src.MDSparsifier import MDSparsifier
from Sparsification_Research.src.SGenerator import SGenerator

import numpy as np

from ..util.eig_functs import euclidean_dist
from ..util import power as pwr

# TODO: tests w/ non-maintaining-diagonal-sparsifier

def sparse_pwr(A, u_0, u_star, num_iter, seed, s):
    """ Power iteration on a sparsified version of A

    Args: 
        A: the original matrix
        u_0: initial guess for top left eigenvector of A
        u_star: actual top left eigenvector of A
        num_iter: number of iterations of power to do
        seed: for duplicatable randomness
        s: degree of sparsification

    Returns:
        xs: a list [0, 1, ..., num_iter - 1]
        ys: the residual per iteration
        lbl: the label associated with this function call (for easy plotting)
    """
    sparsifier = MDSparsifier(seed=seed)

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)


    sparse_A = A.copy()

    sparsifier.sparsify(sparse_A, s)

    v =  pwr.v_from_u(sparse_A, u_0)

    # Initial residual
    xs[0] = 0
    ys[0] = euclidean_dist(u_0, u_star)

    
    for i in range(1, num_iter):
        u, _, v = pwr.topsing(v0=v,
                          A=sparse_A, 
                          maxiter=1)

        # Track x and y
        xs[i] = i
        ys[i] = euclidean_dist(u, u_star)
    
    # print(ys)
    num_mults = pwr.count_mults(A=sparse_A, maxiter=num_iter - 1)
    return xs, ys, f"s = {s:0.6g}, sparsified; {num_mults:,} mults"

def expected_sparse_pwr(A, u_0, u_star, num_iter, seed, x):
    """ Power iteration on a sparsified version of A, expected number of zeroes
    based

    Args: 
        A: the original matrix
        u_0: initial guess for top left eigenvector of A
        u_star: actual top left eigenvector of A
        num_iter: number of iterations of power to do
        seed: for duplicatable randomness
        x: expected number of new zeroes in the sparsified A

    Returns:
        xs: a list [0, 1, ..., num_iter - 1]
        ys: the residual per iteration
        lbl: the label associated with this function call (for easy plotting)
    """

    s_generator = SGenerator(A.shape[0], A.nnz)

    s = s_generator.get_min_s(x)

    xs, ys, lbl = sparse_pwr(A, u_0, u_star, num_iter, seed, s)
    return xs, ys, f"x = {x}, {lbl}"

def percent_sparse_pwr(A, u_0, u_star, num_iter, seed, p):
    """ Power iteration on a sparsified version of A, percentage based

    Args: 
        A: the original matrix
        u_0: initial guess for top left eigenvector of A
        u_star: actual top left eigenvector of A
        num_iter: number of iterations of power to do
        seed: for duplicatable randomness
        p: expected proportion of new zeroes in the sparsified A (not counting
           diagonal entries)

    Returns:
        xs: a numpy array [0, 1, ..., num_iter - 1]
        ys: the residual per iteration
        lbl: the label associated with this function call (for easy plotting)
    """
    s_generator = SGenerator(A.shape[0], A.nnz)

    # get s associated with an expected percent sparsification p
    expected_proportion = p / 100
    s = s_generator.proportion_sparse_s(p=expected_proportion, 
                                        include_diags=False) 

    xs, ys, lbl = sparse_pwr(A, u_0, u_star, num_iter, seed, s)

    return xs, ys, f"{p}%, {lbl}"


def sparse_pwr_tol(A, u_0, u_star, num_iter, seed, s, tolerance=1e-07):
    """ Power iteration on a sparsified version of A, stop after some tolerance 
    is met

    Args: 
        A: the original matrix
        u_0: initial guess for top left eigenvector of A
        u_star: actual top left eigenvector of A
        num_iter: number of iterations of power to do
        seed: for duplicatable randomness
        s: degree of sparsification
        tolerance: determines when top vector is a strong enough approximation

    Returns:
        xs: a list [0, 1, ..., max_iter]
        ys: the residual per iteration
        lbl: the label associated with this function call (for easy plotting)
    """
    sparsifier = MDSparsifier(seed=seed)

    # xs = np.zeros(num_iter)
    # ys = np.zeros(num_iter)
    xs = []
    ys = []

    sparse_A = A.copy()

    sparsifier.sparsify(sparse_A, s)

    v =  pwr.v_from_u(sparse_A, u_0)

    # Initial residual
    # xs[0] = 0
    # ys[0] = euclidean_dist(u_0, u_star)
    xs.append(0)
    ys.append(euclidean_dist(u_0, u_star))

    # Initialize i (number of iterations of power)
    i = 1

    # top singular value is None for first iteration
    s_prev = None

    while (i <  num_iter):
        u, s_curr, v = pwr.topsing(v0=v,
                          A=sparse_A, 
                          maxiter=1)

        # Track x and y
        # xs[i] = i
        # ys[i] = euclidean_dist(u, u_star)

        xs.append(i)
        ys.append(euclidean_dist(u, u_star))

        # Check convergence
        if s_prev is not None and abs(s_curr - s_prev) < tolerance:
            break

        i += 1
        s_prev = s_curr
    
    # print(ys)
    num_mults = pwr.count_mults(A=sparse_A, maxiter=i - 1)
    return xs, ys, f"s = {s:0.6g}, sparsified; {num_mults:,} mults"