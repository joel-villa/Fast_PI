"""
Contains sampling tests, which have y-axis of score of the current 
top vector guess
"""
from .score import baseline

from ..util import subset as sub
from ..util import scikit_jl as jlsk

def col_sample(A, u_0, s_star, max_iter, tol, seed, d, type):
    """ Test of random col-sampling 

    Args:
        A: the matrix to do power iteration on
        u_0: initial guess for top left eigenvector
        s_star: actual top singular value
        max_iter: maximum number of iterations to do power iteration
        tol: how much tolerance (for stopping condition of power iteration)
        seed: for repeatable randomness (doesn't work due to scikit's not 
              supporting this functionality)
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