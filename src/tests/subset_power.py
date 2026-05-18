"""
Some tests for dimensionality reduction where you use random subset of the 
columns to see if SVD converges faster
"""
import numpy as np
from ..util.power import v_from_u
from ..util.power import topsing
from ..util.eig_functs import euclidean_dist
from ..util.scikit_jl import percent_reduce

def select_d_random_columns(A, d, seed):
    """
    A - a matrix (nxm)
    d - the number of columns to get
    seed - for predictable randomness

    RETURN: B - where B is a (nxd) subset of A

    Given a matrix A, get x of its columns randomly
    """

    A_cols = A.shape[1]

    rng = np.random.default_rng(seed=seed)
    cols = rng.choice(A_cols, size=d, replace=False)

    sorted_cols = np.sort(cols)
    
    # TODO: should we be using coo still? 

    # converting to csc for list slicing
    A_csc = A.copy()
    A_csc = A.tocsc()

    # Array slicing
    B = A_csc[:, sorted_cols]

    return B

def subset_svd(A, u_0, u_star, num_iter, seed, d):
    """
    A - the matrix (nxm)
    u_0 - an initial guess for the top left eigenvector of A
    u_star - the actual top left eigenvector of A
    num_iter - the number of iterations of SVD to do
    seed - for repeatable tests
    d - reduction size: A_reduced is (nxd)

    RETURN: xs - a list of iterations [0, 1, 2, ..., num_iter - 1]
            ys - the list of residuals per iteration
    Measure convergence of SVD with some random subset of the columns of A
    """

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)

    ys[0] = euclidean_dist(u_0, u_star)
    xs[0] = 0

    A_reduced = select_d_random_columns(A, d, seed)

    v =  v_from_u(A_reduced, u_0)

    for i in range(1, num_iter):
        # NOTE: using scikit-learn -> top left eig (u) is of significance

        u, _, v = topsing(v0=v,
                          A=A_reduced, 
                          maxiter=1)
        
        euc_dist = euclidean_dist(u, u_star)
        ys[i] = euc_dist
        xs[i] = i

    return xs, ys, f"random column subset: {A_reduced.shape}"

def percent_subset_svd(A, u_0, u_star, num_iter, seed, p):
    """
    A - the matrix (nxm)
    u_0 - an initial guess for the top left eigenvector of A
    u_star - the actual top left eigenvector of A
    num_iter - the number of iterations of SVD to do
    seed - for repeatable tests
    p - reduction percentage: A_reduced is (nx(1-p)*m)

    RETURN: xs - a list of iterations [0, 1, 2, ..., num_iter - 1]
            ys - the list of residuals per iteration
    Measure convergence of SVD with some random subset of the columns of A
    """

    d = percent_reduce(A.shape[1], p)

    xs, ys, label = subset_svd(A, u_0, u_star, num_iter, seed, d)

    return xs, ys, f"{label}, ({p}%)"

def subset_svd_swap(A, u_0, u_star, num_iter, seed, d, step_size):
    """
    A - the matrix (nxm)
    u_0 - an initial guess for the top left eigenvector of A
    u_star - the actual top left eigenvector of A
    num_iter - the number of iterations of SVD to do
    seed - for repeatable tests
    d - reduction size: A_reduced is (nxd)
    step_size - how often to regenerate A

    RETURN: xs - a list of iterations [0, 1, 2, ..., num_iter - 1]
            ys - the list of residuals per iteration
    Measure convergence of SVD with some random subset of the columns of A
    """

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)

    ys[0] = euclidean_dist(u_0, u_star)
    xs[0] = 0

    A_reduced = select_d_random_columns(A, d, seed)

    v =  v_from_u(A_reduced, u_0)

    for i in range(1, num_iter):
        # NOTE: using scikit-learn -> top left eig (u) is of significance
        if (i % step_size == 0):
            # Randomly regenerate A
            reduced_A = select_d_random_columns(A, d=d, seed=seed*i)
            v = v_from_u(reduced_A, u)

        u, _, v = topsing(v0=v,
                          A=A_reduced, 
                          maxiter=1)
        
        euc_dist = euclidean_dist(u, u_star)
        ys[i] = euc_dist
        xs[i] = i

    return xs, ys, f"column subset (swapping every {step_size}): {A_reduced.shape}"

def percent_subset_svd_swap(A, u_0, u_star, num_iter, seed, p, step_size):
    """
    A - the matrix (nxm)
    u_0 - an initial guess for the top left eigenvector of A
    u_star - the actual top left eigenvector of A
    num_iter - the number of iterations of SVD to do
    seed - for repeatable tests
    p - reduction percentage: A_reduced is (nx(1-p)*m)
    step_size - how often to regenerate A

    RETURN: xs - a list of iterations [0, 1, 2, ..., num_iter - 1]
            ys - the list of residuals per iteration
    Measure convergence of SVD with some random subset of the columns of A
    """

    d = percent_reduce(A.shape[1], p)

    xs, ys, label = subset_svd_swap(A, u_0, u_star, num_iter, seed, d, step_size)

    return xs, ys, f"{label}, ({p}%)"