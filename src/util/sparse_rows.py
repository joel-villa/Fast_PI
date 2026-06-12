"""
Contains methods which follow from the proofs in:
    https://www.overleaf.com/project/69729e24436d9a5d68dd5d71

The idea: turn those ROWS with low 2-norm to zero with higher probability
"""

import numpy as np
from scipy.sparse.linalg import norm

def two_norm(A, d, seed):
    """ A selection which involves keeping and scaling the i'th row 
    with probability p_i^2, where p_i = sqrt(s) * ||A^(i)|| / ||A||_F

    The following has been proven to preserve magnitudes of Matrix Vector 
    multiplication in expectation

    PF:
        E[||\tilde Ax||^2] = E[ \tilde Ax dot \tilde Ax ]
                           = E[(\tilde Ax)^T (\tilde Ax)]
                           = E[x^T \tilde A^T \tilde A x]
                           = x^T E[\tilde A^T \tilde A] x
                           = x^T A^T A x
                           = ||Ax||^2
    
    Args: 
        A: a sparse matrix (mxn) in COO format #TODO: could be optimized for CSR format
        d: the expected number of rows to keep
        seed: for predictable randomness

    RETURN: A w/ d nonzero rows in expectation
    """
    root_d = np.sqrt(d)

    # The 2-norm of each row
    row_norms = norm(A, ord=2, axis=1)

    # The probability of keeping each row 
    # (NOTE: numpy `*` operator and `/` are element-wise)
    probabilites = root_d * row_norms / norm(A, ord="fro")

    # The amount to scale up each row
    scales = 1 / np.sqrt(probabilites)

    # Random Number Generator
    rng = np.random.default_rng(seed=seed)

    # Some helper variables
    row_prev = -1
    ith_prob = 0

    # The new matrix
    A_tilde = A.copy()

    # The rows array
    rows = A.coords[0]

    # Remove or keep and scale the i'th row
    for i in range(A.nnz):
        # TODO: could be optimized w/ CSR format
        # print(A.coords[i])
        row_curr = rows[i]

        # Probaility to keep this entry
        prob_keep = probabilites[row_curr]

        if row_curr != row_prev:
            # Update the probability for this row
            ith_prob = rng.random()
        
        if ith_prob < prob_keep:
            # Keep this value and scale it up the associated ammount
            A_tilde.data[i] = A.data[i] * scales[row_curr]

        else:
            # Turn this value to zero
            A_tilde.data[i] = 0
        
        # Update row_prev
        row_prev = row_curr
    
    # Update A to no longer store those new zero rows in memory
    A_tilde.eliminate_zeros()

    return A_tilde