"""
For testing the implementation which puts of JL down projections, to only do
matrix vector multiplications inside power iteration
"""

import numpy as np

from ..util.subset import get_reduce_funct

from ..util import power as pwr
from ..util import score as scr
from ..util import jl_implementation as jl
from ..util import scikit_jl as jlsk

def lazy(A, u_0, s_star, max_iter, tol, seed, d, eps, A_tilde=None):
    """ For testing the implementation which puts of JL down projections, to 
    only do matrix vector multiplications inside power iteration

    Args:
        A: the original matrix
        u_0: initial guess for top left eigenvector
        s_star: actual top singular value
        max_iter: maximum number of iterations to do power iteration
        tol: how much tolerance (for stopping condition of power iteration)
        seed: for repeatable randomness (scikit does not have repeatable 
              randomness)
        d: reduced dimension
        eps: for jl validity checking

    Return:
        xs: an array of amount of scalar mults done
        ys: an array of error of each guess abs(lamba* - lambda) / abs(lambda*)
        lbl: the string representation of this test
    """
    if A_tilde is None:
        A_tilde = A
    s_curr =  pwr.s_from_u(A, u_0)

    if d is None:
        # Set to minimum allowable based on epsilon
        d = jlsk.get_min_dim(X=A_tilde, eps=eps)

    A_reduced = jl.jl_simple(X=A_tilde, d=d, seed=seed, eps=eps) #TODO: hand in v0 rather than u0
    v = pwr.v_from_u(A_reduced, u_0) #TODO: check this 

    xs = np.zeros(max_iter)
    ys = np.zeros(max_iter)

    error_rate = scr.error(s_approx=s_curr, s_star=s_star)
    ys[0] = error_rate
    xs[0] = 0

    m, n = A_tilde.shape

    projection_mat = jl.jl_matrix(n=n,
                                  d=d,
                                  seed=seed,
                                  eps=eps,
                                  )

    s_prev = s_curr

    # print(f"A_tilde.shape: {A_tilde.shape}, P.shape: {projection_mat.shape}, v.shape: {v.shape}")

    for i in range(1, max_iter):
        # NOTE: using scikit-learn -> top left eig (u) is of significance
        u, _, v = pwr.power_lazy(v0=v,
                                 A=A_tilde,
                                 P=projection_mat, 
                                 maxiter=1,
                                 ) #TODO: maybe use this s for convergence? 
        
        # Get score of current approximation of top left eigenvector
        s_curr =  pwr.s_from_u(A, u)
        
        # ERROR RATE: abs(top eigenvalue - approximate eigenvalue) / top eigenvalue
        error_rate = scr.error(s_approx=s_curr, s_star=s_star)

        ys[i] = error_rate

        # Track amount of work done
        xs[i] = xs[i-1] + pwr.count_mults_lazy(v0=v, A=A_tilde, P=projection_mat, maxiter=1)
        
        if (scr.converged(s_curr=s_curr, s_prev=s_prev, tol=tol)):
            # Converged
            break

        # Save current score as previous
        s_prev = s_curr

    _, d = projection_mat.shape

    return xs, ys, f"jl simple (lazy) ({A_reduced.shape[0]}, {d})"

def lazy_percent(A, u_0, s_star, max_iter, tol, seed, p, eps, A_tilde=None):
    """ Percent reduction test

    Args:
        A: the original matrix
        u_0: initial guess for top left eigenvector
        s_star: actual top singular value
        max_iter: maximum number of iterations to do power iteration
        tol: how much tolerance (for stopping condition of power iteration)
        seed: for repeatable randomness (scikit does not have repeatable 
              randomness)
        d: reduced dimension
        eps: for jl validity checking

    Return:
        xs: an array of amount of scalar mults done
        ys: an array of error of each guess abs(lamba* - lambda) / abs(lambda*)
        lbl: the string representation of this test
    """
    if A_tilde is None: #TODO: make all this A_tilde stuff more readable
        A_tilde = A

    _, n = A_tilde.shape

    if p is None:
        d = None
    else:
        d = jlsk.percent_reduce(n=n, p=p)

    xs, ys, lbl = lazy(A=A,
                     u_0=u_0,
                     s_star=s_star,
                     max_iter=max_iter,
                     tol=tol,
                     seed=seed,
                     d=d,
                     eps=eps,
                     A_tilde=A_tilde,
                     )
    
    if d is None:
        return xs, ys, f"eps={eps} {lbl}"
    return xs, ys, f"{p}% {lbl}"


def sample_lazy(A, u_0, s_star, max_iter, tol, seed, p, eps, type):
    """ Sample the columns of A, then do power iteration with a dimensionality 
    reduction on those randomly sampled columns
    
    Args:
        A: the original matrix
        u_0: initial guess for top left eigenvector
        s_star: actual top singular value
        max_iter: maximum number of iterations to do power iteration
        tol: how much tolerance (for stopping condition of power iteration)
        seed: for repeatable randomness (scikit does not have repeatable 
              randomness)
        p: the percentage of columns to sample, and the percentage to then 
        dimensionally reduce #TODO: make this into two separate args
        eps: for jl validity checking
        type: the type of sampling to use
    """

    # COLUMN SAMPLING
    _, n = A.shape
    d = jlsk.percent_reduce(n=n, p=p)
    A_reduced = None
    reduce = get_reduce_funct(type)
    A_reduced = reduce(A=A, d=d, seed=seed)

    # LAZY POWER ITERATION

    xs, ys, lbl = lazy_percent(
        A=A,
        u_0=u_0,
        s_star=s_star,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        p=p,
        eps=eps,
        A_tilde=A_reduced,
    )

    return xs, ys, f"{100 - p}% {type} sample, {lbl}"