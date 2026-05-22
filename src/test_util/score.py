"""
Some helper functions for calculating 'score' of top eigenvector approximations
NOTE: 'score' is shorthand for closeness of top singular value to actual 
"""

def error(s_approx, s_star):
    """ How close is the approximation vector's top singular value to the 
    solution's top singular value (relatively)

    Args: 
        s_approx: guess's score
        s_star: best possible score
    Return: 
        relative error of s_approx
    """
    return abs(s_star - s_approx) / abs(s_star)

def converged(s_curr, s_prev, tol):
    """ Has the power iteration converged? 

    Args:
        s_curr: score of current iteration's approximation
        s_prev: score of previous iteration's  approximation

    Return:
        True if converged, False otw
    """
    return (abs(s_curr - s_prev) / abs(s_curr)) < tol