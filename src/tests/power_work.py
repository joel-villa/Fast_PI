"""
For functions which plot work (number of scalar mults) on the x-axis, and 
difference in top singular value on the y-axis
"""

import numpy as np

from ..util import power as pwr
from ..util import score as scr


def baseline_pwr_s_based(A, u_0, s_star, num_iter, tol, seed):
    """ The baseline power work-load (number of scalar mults)

    Args:
        A: the matrix to do power iteration on
        u_0: initial guess for top left eigenvector
        s_star: actual top singular value
        num_iter: number of iterations to do power
        tol: how much tolerance? 
        seed: for repeatable randomness (doesn't work due to scikit's not 
              supporting this functionality)
    Return:
        xs: an array [0, 1, 2, ..., num_iter - 1]
        ys: an array of residual of each guess for u (top left eigencector)
        lbl: the string representation of this test
    """

    s_curr =  pwr.s_from_u(A, u_0)
    v =  pwr.v_from_u(A, u_0)

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)

    ys[0] = scr.error(s_approx=s_curr, s_star=s_star)
    xs[0] = 0

    s_prev = s_curr

    for i in range(1, num_iter):
        # NOTE: using scikit-learn -> top left eig (u) is of significance
        u, s_curr, v = pwr.topsing(v0=v,
                                   A=A, 
                                   maxiter=1,
                                   )
        
        # ERROR RATE: abs(top eigenvalue - approximate eigenvalue) / top eigenvalue
        error_rate = scr.error(s_approx=s_curr, s_star=s_star)

        ys[i] = error_rate

        # Track amount of work done
        xs[i] = xs[i-1] + pwr.count_mults(A=A, maxiter=1)
        
        if ((abs(s_curr - s_prev) / abs(s_curr)) < tol):
            # Converged
            break

        # Save current score as previous
        s_prev = s_curr
    
    return xs, ys, f"standard power {A.shape}" #TODO: Test this baddie