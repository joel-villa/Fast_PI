import numpy as np

from . import s_funcs as func

def pi_scale_pi_squared_select(A, scale_func, sf_kwargs, seed):
    """ A selection which involves keeping and scaling the i'th row by a factor
    of p_i, with probability p_i^2, where p_i is a function input by user

    The following has been proven to preserve magnitudes of Matrix Vector 
    multiplication in expectation

    PF:
        ||\tilde Ax||^2 = \sum_{i = 1}^n \sum_{j = 1}^n x_i x_j (\tilde A^T \tilde A)_{ij}
                        = \sum_{i = 1}^n \sum_{j = 1}^n x_i x_j ((A^T A)_{ij} + \sum_{k=1}^n \frac{X_k - p_k^2}{p_k^2}A_{ki}A_{kj})
                        = \sum_{i = 1}^n \sum_{j = 1}^n x_i x_j (A^TA)_{ij} + \sum_{i = 1}^n \sum_{j = 1}^n \sum_{k=1}^n \frac{X_k - p_k^2}{p_k^2} x_i x_jA_{ki}A_{kj}
                        = || Ax||^2 + \sum_{i=1}^n \frac{X_i - p_i^2}{p_i^2} (A^{(i)}x)^2
    
    Args: 
        A: a sparse matrix (mxn) in COO format 
        scale_func: How to upscale the rows (Note the scale is the sqrt of the 
                    probability)
        sf_kwargs: Those arguments to pass to the probability function
        seed: for predictable randomness

    RETURN:
        ~A s.t. all those rows of A have been scaled or turned to zero based on 
        the given prob_func
    """

    #Reading the info from A, to determine weights of each row
    A = A.tocsr(copy=False)

    scales = scale_func(A, **sf_kwargs)

    # Random Number Generator
    rng = np.random.default_rng(seed=seed)

    # The new matrix
    A_tilde = A.copy()

    # The scaling factors, and probabilities of keeping each row
    probabilities = scales * scales

    if (probabilities.max() > 1):
        raise ValueError(f"probabilities.max() = {probabilities.max()} > 1") 

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

    print(f"{scale_func.__name__} kept_rows: {kept_rows}, new zeros: {A.nnz - A_tilde.nnz}")
    #TODO: print expected number of rows kept

    return A_tilde


def get_A_tilde(A, gen_type, sf_kwargs, seed):
    """ get a altered version of A, given some type of modification

    Args:
        A: a sparse matrix (mxn) in COO format 
        gen_type: Determines the scaling function to use
        sf_kwargs: Those arguments to pass to the probability function
        seed: for predictable randomness
    """
    scale_func = None

    # Get scaling function
    match gen_type:
        case "one-norm":
            scale_func = func.norm_max_based
            sf_kwargs = sf_kwargs | {"norm_ord": 1}
        case "two-norm":
            scale_func = func.norm_max_based
            sf_kwargs = sf_kwargs | {"norm_ord": 2}
        case "inf-norm":
            scale_func = func.norm_max_based
            sf_kwargs = sf_kwargs | {"norm_ord": np.inf}
        case "uniform":
            scale_func = func.uniform
        case _:
            raise ValueError(f"gen_type = {gen_type}, not an implemented method")
        
    return pi_scale_pi_squared_select(
        A=A,
        scale_func=scale_func,
        sf_kwargs=sf_kwargs,
        seed=seed,
    )