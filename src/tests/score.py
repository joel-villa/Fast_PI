"""
For functions which plot work (number of scalar mults) on the x-axis, and 
difference in top singular value on the y-axis
"""

import numpy as np

from ..util.subset import select_d_random_columns

from ..util import power as pwr
from ..util import score as scr
from ..util import jl_implementation as jl
from ..util import scikit_jl as jlsk


def baseline(A, u_0, s_star, max_iter, tol, init_mults=0, A_tilde=None):
    """ The baseline power work-load (number of scalar mults)

    Args:
        A: the original matrix
        u_0: initial guess for top left eigenvector
        s_star: actual top singular value
        max_iter: maximum number of iterations to do power iteration
        tol: how much tolerance (for stopping condition of power iteration)
        seed: for repeatable randomness (scikit does not have repeatable 
              randomness)
        init_mults: the initial number of scalar mults
        A_tilde: some augmented version of A (more sparse, less dimensions, etc)

    Return:
        xs: an array of amount of scalar mults done
        ys: an array of error of each guess abs(lamba* - lambda) / abs(lambda*)
        lbl: the string representation of this test
    """

    if A_tilde is None:
        # Baseline case, no augmentation
        A_tilde = A

    s_curr =  pwr.s_from_u(A, u_0)
    v =  pwr.v_from_u(A_tilde, u_0)

    xs = np.zeros(max_iter)
    ys = np.zeros(max_iter)

    error_rate = scr.error(s_approx=s_curr, s_star=s_star)
    ys[0] = error_rate
    xs[0] = init_mults

    s_prev = s_curr

    for i in range(1, max_iter):
        # NOTE: using scikit-learn -> top left eig (u) is of significance
        u, _, v = pwr.topsing(v0=v,
                              A=A_tilde, 
                              maxiter=1,
                              ) #TODO: maybe use this s for convergence? 
        
        # Get score of current approximation of top left eigenvector
        s_curr =  pwr.s_from_u(A, u)
        
        # ERROR RATE: abs(top eigenvalue - approximate eigenvalue) / top eigenvalue
        error_rate = scr.error(s_approx=s_curr, s_star=s_star)

        ys[i] = error_rate

        # Track amount of work done
        xs[i] = xs[i-1] + pwr.count_mults(v0=v, A=A_tilde, maxiter=1)
        
        if (scr.converged(s_curr=s_curr, s_prev=s_prev, tol=tol)):
            # Converged
            break

        # Save current score as previous
        s_prev = s_curr

    return xs, ys, f"baseline {A.shape}"

def jl_dimension(A, u_0, s_star, max_iter, tol, seed, d, eps, type):
    """ Test of jl implementations

    Args:
        A: the matrix to do power iteration on
        u_0: initial guess for top left eigenvector
        s_star: actual top singular value
        max_iter: maximum number of iterations to do power iteration
        tol: how much tolerance (for stopping condition of power iteration)
        seed: for repeatable randomness (doesn't work due to scikit's not 
              supporting this functionality)
        d: reduced dimension
        eps: for jl-reduction
        type: string representation of reduction type
    Return:
        xs: an array of amount of scalar mults done
        ys: an array of error of each guess abs(lamba* - lambda) / abs(lambda*)
        lbl: the string representation of this test
    """

    jl_funct = None

    match type.lower():
        # Get reduction function
        case "simple":
            jl_funct = jl.jl_simple
        case "gaussian":
            jl_funct = jlsk.jl_gaussian
        case "sparse":
            jl_funct = jlsk.jl_sparse
        case _:
            raise TypeError(f"Invalid reduction type: {type}")

    A_reduced = jl_funct(X=A, d=d, seed=seed, eps=eps)
    
    init_mults = jlsk.reduction_cost(X=A, d=d)

    xs, ys, _ = baseline(A=A,
                         A_tilde=A_reduced,
                         u_0=u_0,
                         s_star=s_star,
                         max_iter=max_iter,
                         tol=tol,
                         init_mults=init_mults)

    return xs, ys, f"jl {type}, {A_reduced.shape}"

def jl_percent(A, u_0, s_star, max_iter, tol, seed, p, eps, type):
    """ Test of jl implementations with percent reductions

    Args:
        A: the matrix to do power iteration on
        u_0: initial guess for top left eigenvector
        s_star: actual top singular value
        max_iter: maximum number of iterations to do power iteration
        tol: how much tolerance (for stopping condition of power iteration)
        seed: for repeatable randomness (doesn't work due to scikit's not 
              supporting this functionality)
        p: reduction percentage
        eps: for jl-reduction
        type: string representation of reduction type
    Return:
        xs: an array of amount of scalar mults done
        ys: an array of error of each guess abs(lamba* - lambda) / abs(lambda*)
        lbl: the string representation of this test
    """
    _, n = A.shape

    d = jlsk.percent_reduce(n=n, p=p)

    xs, ys, lbl = jl_dimension(A=A,
                             u_0=u_0,
                             s_star=s_star,
                             max_iter=max_iter,
                             tol=tol,
                             seed=seed,
                             d=d,
                             eps=eps,
                             type=type,
                             )

    return xs, ys, f"{p}% {lbl}"

def row_sample(A, u_0, s_star, max_iter, tol, seed, d):
    """ Test of random row-sampling 

    Args:
        A: the matrix to do power iteration on
        u_0: initial guess for top left eigenvector
        s_star: actual top singular value
        max_iter: maximum number of iterations to do power iteration
        tol: how much tolerance (for stopping condition of power iteration)
        seed: for repeatable randomness (doesn't work due to scikit's not 
              supporting this functionality)
        d: reduced dimension
    Return:
        xs: an array of amount of scalar mults done
        ys: an array of error of each guess abs(lamba* - lambda) / abs(lambda*)
        lbl: the string representation of this test
    """

    A_reduced = select_d_random_columns(A, d, seed)

    xs, ys, _ = baseline(A=A,
                         A_tilde=A_reduced,
                         u_0=u_0,
                         s_star=s_star,
                         max_iter=max_iter,
                         tol=tol,
                         init_mults=0)

    return xs, ys, f"row sample, {A_reduced.shape}"

def row_sample_p(A, u_0, s_star, max_iter, tol, seed, p):
    """ Test of random row sampling based on percent of reduction

    Args:
        A: the matrix to do power iteration on
        u_0: initial guess for top left eigenvector
        s_star: actual top singular value
        max_iter: maximum number of iterations to do power iteration
        tol: how much tolerance (for stopping condition of power iteration)
        seed: for repeatable randomness (doesn't work due to scikit's not 
              supporting this functionality)
        p: reduction percentage
        eps: for jl-reduction
        type: string representation of reduction type
    Return:
        xs: an array of amount of scalar mults done
        ys: an array of error of each guess abs(lamba* - lambda) / abs(lambda*)
        lbl: the string representation of this test
    """
    _, n = A.shape

    d = jlsk.percent_reduce(n=n, p=p)

    xs, ys, lbl = row_sample(A=A,
                             u_0=u_0,
                             s_star=s_star,
                             max_iter=max_iter,
                             tol=tol,
                             seed=seed,
                             d=d,
                             )

    return xs, ys, f"{p}% {lbl}"