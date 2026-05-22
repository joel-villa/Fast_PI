"""
Some helper functions for generating plots of tests 
(called by those files in th esrc directory)
"""

import numpy as np

from Sparsification_Research.src.SSGetter import SSGetter

import numpy as np
from scipy.linalg import norm # 2-norm by default
from ..test_util import eig_functs as eigs



def init(mat_name, seed):
    """ Get intial information for tests

    Args: 
        mat_name: the name of the matrix in the SuiteSparse Matrix Collection
        seed: for generating initial guess for top left eigenvector

    Return:
        A: the matrix in CSR format
        kwargs: a dictionary containing the initial guess for u, as wall as the 
                solution u
    """
    ss_getter = SSGetter(in_csr=False)
    A = ss_getter.get(mat_name)
        
    u_star =  eigs.top_left(A)

    rng = np.random.default_rng(seed=seed)
    u0 = rng.normal(0, 1, A.shape[0])
    u0 = u0 / norm(u0)

    print(f"Testing {mat_name}")

    kwargs = {"u_star": u_star,
              "u_0": u0,
              }

    return A, kwargs
