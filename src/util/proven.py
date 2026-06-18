import numpy as np

def two_norm(A, prob_func, seed):
    """ A selection which involves keeping and scaling the i'th row by a factor
    of p_i, with probability p_i^2, where p_i is a function input by user

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
        A: a sparse matrix (mxn) in COO format 
        prob_func: How to weight the rows
        seed: for predictable randomness

    RETURN:
        ~A s.t. all those rows of A have been scaled or turned to zero based on 
        the given prob_func
    """

    #Reading the info from A, to determine weights of each row
    A = A.tocsr(copy=False)

    # A probability per row
    probabilities = np.zeros(A.shape[0])

    # Compute p_i per each row
    for row in range(A.shape[0]):
        # Where this row's data starts and ends
        start = A.indptr[row]
        end = A.indptr[row + 1]

        row_data = A.data[start:end]

        # Probability to keep and scale this row
        probabilities[row] = prob_func(row_data)
        
        pi_squared = probabilities[row] * probabilities[row]
        if (pi_squared < 0 or pi_squared > 1):
            raise ValueError(f"pi^2 = {pi_squared}, pi^2 needs to be between 0 and 1 for all i")

    # Random Number Generator
    rng = np.random.default_rng(seed=seed)

    # The new matrix
    A_tilde = A.copy()

    # The scaling factors, and probabilities of keeping each row
    scales = probabilities
    probabilities = probabilities * probabilities

    kept_rows = 0

    # Remove or keep and scale the i'th row
    for row in range(A_tilde.shape[0]):
        # Where this row's data starts and ends
        start = A_tilde.indptr[row]
        end = A_tilde.indptr[row + 1]

        # Probability to keep and scale this row
        prob_keep = probabilities[row]

        keep = rng.random() < prob_keep

        row_data = A_tilde.data[start:end]

        if keep:
            kept_rows += 1
            A_tilde.data[start:end] = row_data * scales[row]
        else:
            A_tilde.data[start:end] = np.zeros_like(row_data)
    
    # Update A to no longer store those new zero rows in memory
    A_tilde.eliminate_zeros()

    print(f"{prob_func.__name__} kept_rows: {kept_rows}, new zeros: {A.nnz - A_tilde.nnz}")

    return A_tilde