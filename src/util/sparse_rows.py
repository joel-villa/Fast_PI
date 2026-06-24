"""
Contains methods which follow from the proofs in:
    https://www.overleaf.com/project/69729e24436d9a5d68dd5d71

The idea: turn those ROWS with low 2-norm to zero with higher probability
"""

import numpy as np

from scipy.sparse.linalg import norm

from . import subset as util

def calc_row_norms(A, ord):
    """ Calculate the row norms of a matrix

    Args: 
        A: the matrix (scipy.sparse format)
        ord: what order of norm (1, 2, inf, 'fro', ...)
    Return: norm of all the rows of A
    """
    return norm(A, ord=ord, axis=1)

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
    BUG: p_i^2 exceeds 1, meaning some rows have all the probability density, 
         others have little -> behaves poorly
    """
    #TODO, must use CSR to prevent independence within rows
    root_d = np.sqrt(d)

    # The 2-norm of each row
    row_norms = calc_row_norms(A=A, ord=2)

    fro_norm = np.sqrt(np.sum(row_norms * row_norms))
    fro_norm = norm(A, ord="fro")

    # The sqrt of the probabiility of keeping each row
    # (NOTE: numpy `*` operator and `/` are element-wise)
    probabilites = root_d * row_norms / fro_norm

    probabilites = np.minimum(probabilites, 1)

    # The amount to scale up each row
    scales = 1 / probabilites

    # The probability of keeping each row
    probabilites = probabilites * probabilites

    # Random Number Generator
    rng = np.random.default_rng(seed=seed)

    # Some helper variables
    row_prev = -1
    ith_prob = 0

    # The new matrix
    A_tilde = A.tocsr(copy=True)

    # The rows array
    rows = A.coords[0]

    kept_rows = 0

    print("rows:", A.shape[0])
    print("sum p_i^2 =", probabilites.sum())

    # Remove or keep and scale the i'th row
    for row in range(A_tilde.shape[0]):
        # Where this row's data starts and ends
        start = A_tilde.indptr[row]
        end = A_tilde.indptr[row + 1]

        # Probability to keep and scale this row
        prob_keep = probabilites[row]

        keep = rng.random() < prob_keep

        row_data = A_tilde.data[start:end]

        if keep:
            kept_rows += 1
            A_tilde.data[start:end] = row_data * scales[row]
        else:
            A_tilde.data[start:end] = np.zeros_like(row_data)
    
    # Update A to no longer store those new zero rows in memory
    A_tilde.eliminate_zeros()

    print(f"d = {d}, kept_rows: {kept_rows}, new zeros: {A.nnz - A_tilde.nnz}, A.nnz = {A.nnz}, A_tilde.nnz = {A_tilde.nnz}")

    return A_tilde

    # # Remove or keep and scale the i'th row
    # for i in range(A.nnz):
    #     # TODO: could be optimized w/ CSR format
    #     # print(A.coords[i])
    #     row_curr = rows[i]

    #     # Probaility to keep this entry
    #     prob_keep = probabilites[row_curr]

    #     if row_curr != row_prev:
    #         # Update the probability for this row
    #         ith_prob = rng.random()
    #         if ith_prob < prob_keep:
    #             sparsed_rows += 1
        
    #     if ith_prob < prob_keep:
    #         # Keep this value and scale it up the associated ammount
    #         A_tilde.data[i] = A.data[i] * scales[row_curr]

    #     else:
    #         # Turn this value to zero
    #         A_tilde.data[i] = 0
        
    #     # Update row_prev
    #     row_prev = row_curr
    
    # # Update A to no longer store those new zero rows in memory
    # A_tilde.eliminate_zeros()

    # print(f"d = {d}, sparsed_rows: {sparsed_rows}, new zeros: {A.nnz - A_tilde.nnz}")

    # return A_tilde

def two_norm_choice(A, d, seed):
    """ Keep and scale d of those rows of A, using np.choice

    TODO: Problem with this implementation: since using np.choice, don't know the 
    exact p_i to divide the i'th row of A by..

    Args: 
        A: a matrix (mxn)
        d: the number of columns to get
        seed: for predictable randomness

    RETURN: A w/ d nonzero rows
    (TODO: prove for selection w/o replacement)
    """

    # element-wise square of A (NOTE: CSR mats do not support np.square)
    square_A = A.multiply(A)

    # The square of the two norm of the row of A: ||A^(i)||^2
    square_of_norm = np.asarray(np.sum(square_A, axis=1)).ravel() 

    # # The fourth power of the two norm of the columns of A: ||A^(i)||^4
    # fourth_of_norm = np.square(square_of_norm) 

    # # Probability distribution
    # weights = fourth_of_norm / np.sum(fourth_of_norm) 

    # Probability distribution
    weights = square_of_norm / np.sum(square_of_norm) 

    # Those rows to keep and scale
    rows = util.weighted_select(
        n=A.shape[1], 
        d=d, 
        seed=seed, 
        weights=weights
    )

    # Those significant weights
    weights_sig = weights[rows]

    # Scaling distribution
    scales = np.zeros_like(weights)
    scales[rows] = 1 / np.sqrt(weights_sig)

    # Copy of A, to update
    A_tilde = A.copy()

    # The rows array
    rows = A.coords[0]

    # Remove or keep and scale the i'th row
    for i in range(A.nnz):
        # TODO: could be optimized w/ CSR format
        row = rows[i]

        # Update A_tilde
        A_tilde.data[i] = A.data[i] * scales[row]
    
    # Update A to no longer store those new zero rows in memory
    A_tilde.eliminate_zeros()
     
    print(f"d = {d}, new zeros: {A.nnz - A_tilde.nnz}")

    return A_tilde

# TODO: 1-norm? 
# TODO: row reduction, rather than keeping those zero rows
# TODO: maybe I'm testing the wrong thing (xA rather than Ax) in terms of vector
#       scores

def ind_sparse(A, d, seed):
    """ A selection which involves keeping and scaling the j'th element in the 
    i'th row with probability p_i^2, where p_i = sqrt(s) * ||A^(i)|| / ||A||_F
    
    Args: 
        A: a sparse matrix (mxn) in COO format #TODO: could be optimized for CSR format
        d: the expected number of rows to keep
        seed: for predictable randomness

    RETURN: A w/ d nonzero rows in expectation
    TODO: version which each entry is independent, then could use the articles: 
        - The eigenvalues of random symmetric matrices
        - https://dl.acm.org/doi/pdf/10.1145/1219092.1219097
    """
    root_d = np.sqrt(d)

    # The 2-norm of each row
    row_norms = norm(A, ord=2, axis=1)

    # The probability of keeping each row 
    # (NOTE: numpy `*` operator and `/` are element-wise)
    probabilites = root_d * row_norms / norm(A, ord="fro")

    probabilites = np.minimum(probabilites, 1)

    # The amount to scale up each row
    scales = 1 / probabilites

    # Square the probabilities
    probabilites = probabilites * probabilites

    # Random Number Generator
    rng = np.random.default_rng(seed=seed)

    # The new matrix
    A_tilde = A.copy()

    # The rows array
    rows = A.coords[0]

    # Remove or keep and scale the i'th row
    for i in range(A.nnz):
        # TODO: could be optimized w/ CSR format
        # print(A.coords[i])
        row = rows[i]

        # Probaility to keep this entry
        prob_keep = probabilites[row]
        
        if rng.random() < prob_keep:
            # Keep this value and scale it up the associated ammount
            A_tilde.data[i] = A.data[i] * scales[row]

        else:
            # Turn this value to zero
            A_tilde.data[i] = 0
    
    # Update A to no longer store those new zero rows in memory
    A_tilde.eliminate_zeros()

    print(f"d = {d} new zeros: {A.nnz - A_tilde.nnz}")

    return A_tilde