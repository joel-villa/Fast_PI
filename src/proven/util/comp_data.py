"""A file for computing the metadata of theoretical significance of a given 
sparse matrix"""

import math

import numpy as np

import scipy
from scipy.sparse.linalg import norm, eigs

from ..preprocess import preprocess

from ...util.row_norms import get_sorted_row_norms, get_power_law_coefficients

N_STRING:str = "n"
C_STRING:str = "c"
K_STRING:str = "k"
KAPPA_STRING:str = "kappa"
NORM_STRING:str = "op_norm"
VAR_STRING:str = "var_proxy"
SCALE_STRING:str = "scale_factor"
LAMBDA_STRING:str = "lambda"

META_DATA:set[str] = { 
    N_STRING,
    C_STRING, 
    K_STRING, 
    KAPPA_STRING, 
    NORM_STRING,
    VAR_STRING,
    SCALE_STRING,
    LAMBDA_STRING,
}


def get_c_k(mat_name:str) -> tuple[float, float]:
    """Get the power-law-distribution coefficients corresponding to A's row
    magnitude distribution

    Args:
        mat_name (str): The matrix under question

    Returns:
        tuple[float, float]: 
            c: y-intercept of power-law distribution
            k: slope of power-law distribution
    """
    row_norms = get_sorted_row_norms(mat_name=mat_name, rescale=True)

    return get_power_law_coefficients(ys=row_norms)
    

def get_kappa(k:float) -> float:
    """Get kappa from k 

    Args:
        k (float): the slope of the power-law distribution on a log-log plot

    Returns:
        float: -3k - 1
    """
    return -3*k - 1

def get_op_norm(A:scipy.sparse) -> float:
    """Get the operator norm of the sparse matrix

    Args:
        A (scipy.sparse): the matrix

    Returns:
        float: ||A||_2
    """
    return norm(A, ord=2)

def get_eps_net_var_proxy(A:scipy.sparse) -> float:
    """The variance proxy from Matrix Bernstein's + Epsilon net approach

    Args:
        A (scipy.sparse): The matrix in question

    Returns:
        float: || sum_{i=1}^n (||a_i|| - ||a_i||^2)a_i^Ta_i||
    """

    sum_mat = np.zeros(shape=A.shape)

    for i in range(A.shape[0]):
        # Loop through the rows
        row_i = A[i, :] 

        # Compute row magnitudes
        i_vals = row_i.data
        norm_squared = np.sum(i_vals * i_vals) # sum of the square of the values
        norm_i = math.sqrt(norm_squared)

        if norm_i > 1.000001:
            # All rows should have row magnitudes less than or equal to 1
            raise ValueError(f"i'th row magnitude is {norm_i}, {norm_i} > 1")

        scalar_factor = norm_i * (1 - norm_i) 
        outer_prod = row_i.T @ row_i

        sum_mat += scalar_factor * outer_prod

    var_proxy = np.linalg.norm(sum_mat, ord=2)

    if var_proxy > A.shape[0] * 0.25000001:
        # something went wrong: x-x^2 <= 0.25
        raise ValueError(f"var_proxy = {var_proxy}")

    return float(var_proxy)

def get_lambda_v(A:scipy.sparse) -> tuple[float, np.ndarray]:
    """Get the top eigenvector eigenvalue pair of the matrix A

    Args:
        A (scipy.sparse): The matrix in question

    Returns:
        tuple[float, np.ndarray]: [top lambda value, top eigenvector], s.t. 
        A * v = lambda * v, and lambda is the largest eigenvalue in absolute 
        value (i.e. the solution to power iteration)
    """
    top_value, top_vector = eigs(
        A=A,
        k=1, # only require top eigenvalue
        which='LM', #Largest magnitude
    )

    return top_value[0], top_vector.flatten()

def get_eig_vect(mat_name:str) -> np.ndarray:
    """Get the top eigenvector of the given SS matrix

    Args:
        mat_name (str): Suite Sparse Matrix name

    Returns:
        np.ndarray: The top eigenvector
    """
    A, _ = preprocess(mat_name)
    _, eig_vect = get_lambda_v(A)
    return eig_vect

def get_metadata(
        mat_name: str,
) -> dict[str, int | float]:
    """Get the metadata of theoretical signficance for the provided matrix

    Args:
        mat_name (str): the matrix name (suite-sparse)
        A (scipy.sparse): matrix under question

    Returns:
        dict[str, int | float]: 
            "n": number of rows in A
            "c": y-intercept of power-law distribution
            "k": slope of power-law distribution
            "kappa": -3k - 1 (always positive)
            "op_norm": ||A||_2
            "var_proxy": variance proxy from matrix bernsteins
            "scale_factor": how much the matrix is scaled to get ||a_i|| <= 1
    """
    A, scale_factor = preprocess(mat_name)
    n = A.shape[0]
    c, k = get_c_k(mat_name)
    kappa = get_kappa(k)
    op_norm = get_op_norm(A)
    var_proxy = get_eps_net_var_proxy(A)

    eig_val, eig_vect = get_lambda_v(A) #TODO: test this bad boy!

    return {
        N_STRING: n,
        C_STRING: c,
        K_STRING: k,
        KAPPA_STRING: kappa,
        NORM_STRING: op_norm,
        VAR_STRING: var_proxy,
        SCALE_STRING: scale_factor,
        LAMBDA_STRING: eig_val,
    }

if __name__ == '__main__':
    """Main for testing purposes
    """
    mats = [
        "1138_bus",
        "494_bus",
        "Harvard500",
        "bcspwr06",
        "bcsstk07",
        "bcsstk08",
        "bcsstk19",
        "bcsstk34",
        "bcsstm07",
        "blckhole",
        "bp_0",
        "cage7",
        "can_229",
        "dwt_193",
        "eris1176",
        "ex2",
        "fs_541_1",
        "gre_1107",
        "gre_343",
        "hor_131",
        "impcol_d",
        "lshp1561",
        "msc00726",
        "nasa1824",
        "nos3",
        "tomography",
    ]

    mats = sorted(mats) #Alphabetical order

    for mat in mats:
        print(mat)
        get_metadata(mat)