"""
Contains sampling tests, which have y-axis of score of the current 
top vector guess
"""

import numpy as np

from .score import baseline

from ..util import subset as sub
from ..util import power as pwr
from ..util import score as scr
from ..util import scikit_jl as jl

def col_sample(A, u_0, s_star, max_iter, tol, seed, d, type):
    """ Test of random col-sampling 

    Args:
        A: the matrix to do power iteration on
        u_0: initial guess for top left eigenvector
        s_star: actual top singular value
        max_iter: maximum number of iterations to do power iteration
        tol: how much tolerance (for stopping condition of power iteration)
        seed: for repeatable randomness
        d: reduced dimension
        type: string representation of sampling type
    Return:
        xs: an array of amount of scalar mults done
        ys: an array of error of each guess abs(lamba* - lambda) / abs(lambda*)
        lbl: the string representation of this test
    """

    A_reduced = None

    match type.lower():
        case "simple":
            # simple random sampling of columns
            A_reduced = sub.select_d_random_columns(A, d, seed)
        case "1-norm":
            # random sampling of columns, with weight based on 1-norm of column
            A_reduced = sub.one_norm_select(A, d, seed)
        case _:
            raise TypeError(f"Invalid sampling type: {type}")

    xs, ys, _ = baseline(A=A,
                         A_tilde=A_reduced,
                         u_0=u_0,
                         s_star=s_star,
                         max_iter=max_iter,
                         tol=tol,
                         init_mults=0)

    return xs, ys, f"col sample ({type}), {A_reduced.shape}"

def col_sample_p(A, u_0, s_star, max_iter, tol, seed, p, type):
    """ Test of random col sampling based on percent of reduction

    Args:
        A: the matrix to do power iteration on
        u_0: initial guess for top left eigenvector
        s_star: actual top singular value
        max_iter: maximum number of iterations to do power iteration
        tol: how much tolerance (for stopping condition of power iteration)
        seed: for repeatable randomness
        p: reduction percentage
        eps: for jl-reduction
        type: string representation of reduction type
    Return:
        xs: an array of amount of scalar mults done
        ys: an array of error of each guess abs(lamba* - lambda) / abs(lambda*)
        lbl: the string representation of this test
    """
    _, n = A.shape

    d = jl.percent_reduce(n=n, p=p)

    xs, ys, lbl = col_sample(A=A,
                             u_0=u_0,
                             s_star=s_star,
                             max_iter=max_iter,
                             tol=tol,
                             seed=seed,
                             d=d,
                             type=type,
                             )

    return xs, ys, f"{100 - p}% {lbl}"
def reduce_A(A, p, seed, type):
    """ Reduce A based on percent reduction and reduction function
    
    Args:
        A: the matrix to reduce
        p: reduction percentage
        seed: for repeatable randomness
        type: the type of reduction to do (string representation)
    
    Return: the reduced A
    """
    # Get reduction function
    match type.lower():
        case "simple":
            # simple random sampling of columns
            reduce = sub.select_d_random_columns
        case "1-norm":
            # random sampling of columns, with weight based on 1-norm of column
            reduce =  sub.one_norm_select
        case _:
            raise TypeError(f"Invalid sampling type: {type}")
        
    _, n = A.shape

    d = jl.percent_reduce(n=n, p=p)
    return reduce(A, d, seed)

def col_sample_inc_p(A, u_0, s_star, max_iter, tol, seed, p0, type, step, inc_funct):
    """ Test of random col sampling with an increasing percentage of reduction

    Args:
        A: the matrix to do power iteration on
        u_0: initial guess for top left eigenvector
        s_star: actual top singular value
        max_iter: maximum number of iterations to do power iteration
        tol: how much tolerance (for stopping condition of power iteration)
        seed: for repeatable randomness (doesn't work due to scikit's not 
              supporting this functionality)
        p0: initial reduction percentage
        eps: for jl-reduction
        type: string representation of reduction type
        step: how often to increase percentage of reduction 
        inc_funct: function to increase the reduction percentage
    Return:
        xs: an array of amount of scalar mults done
        ys: an array of error of each guess abs(lamba* - lambda) / abs(lambda*)
        lbl: the string representation of this test
    """
    
        
    # Reduce A initially
    p = p0
    

    s_curr =  pwr.s_from_u(A, u_0)
    v =  pwr.v_from_u(A_tilde, u_0)

    xs = np.zeros(max_iter)
    ys = np.zeros(max_iter)

    error_rate = scr.error(s_approx=s_curr, s_star=s_star)
    ys[0] = error_rate
    xs[0] = 0 # Initially no scalar mults done

    s_prev = s_curr

    for i in range(1, max_iter):
        if (i % step == 0):
            # Increase reduction percentage
            p = inc_funct(p)

            # Reduce A again with new reduction percentage
            d = jl.percent_reduce(n=n, p=p)
            A_tilde = reduce(A, d, seed)
        
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


    return xs, ys, f"{100 - p}%, increasing every {step} iterations: {lbl}"