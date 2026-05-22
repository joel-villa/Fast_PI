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