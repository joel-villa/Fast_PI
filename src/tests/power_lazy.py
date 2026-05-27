"""
For testing the implementation which puts of JL down projections, to only do
matrix vector multiplications inside power iteration
"""

import numpy as np

from ..util.subset import select_d_random_columns

from ..util import power as pwr
from ..util import score as scr
from ..util import jl_implementation as jl
from ..util import scikit_jl as jlsk

def lazy(A, u_0, s_star, max_iter, tol, seed, d, eps):
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


    s_curr =  pwr.s_from_u(A, u_0)

    if d is None:
        # Set to minimum allowable based on epsilon
        d = jlsk.get_min_dim(X=A, eps=eps)

    A_reduced = jl.jl_simple(X=A, d=d, seed=seed, eps=eps) #TODO: hand in v0 rather than u0
    v = pwr.v_from_u(A_reduced, u_0) #TODO: check this 

    xs = np.zeros(max_iter)
    ys = np.zeros(max_iter)

    error_rate = scr.error(s_approx=s_curr, s_star=s_star)
    ys[0] = error_rate
    xs[0] = 0

    m, n = A.shape

    projection_mat = jl.jl_matrix(n=n,
                                  d=d,
                                  seed=seed,
                                  eps=eps,
                                  )
    
    
    # jlsk.check_valid_dimensions(A)
    # jlsk.check_safe(A, d, eps)

    s_prev = s_curr

    for i in range(1, max_iter):
        # NOTE: using scikit-learn -> top left eig (u) is of significance
        u, _, v = pwr.power_lazy(v0=v,
                                 A=A,
                                 P=projection_mat, 
                                 maxiter=1,
                                 ) #TODO: maybe use this s for convergence? 
        
        # Get score of current approximation of top left eigenvector
        s_curr =  pwr.s_from_u(A, u)
        
        # ERROR RATE: abs(top eigenvalue - approximate eigenvalue) / top eigenvalue
        error_rate = scr.error(s_approx=s_curr, s_star=s_star)

        ys[i] = error_rate

        # Track amount of work done
        xs[i] = xs[i-1] + pwr.count_mults_lazy(v0=v, A=A, P=projection_mat, maxiter=1)
        
        if (scr.converged(s_curr=s_curr, s_prev=s_prev, tol=tol)):
            # Converged
            break

        # Save current score as previous
        s_prev = s_curr

    _, d = projection_mat.shape

    return xs, ys, f"jl simple (lazy) ({A.shape[0]}, {d})"

def lazy_percent(A, u_0, s_star, max_iter, tol, seed, p, eps):
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
    _, n = A.shape

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
                     eps=eps)

    return xs, ys, f"{p}% {lbl}"