"""A file for getting the metadata of thoeretical significance"""

import scipy

from Sparsification_Research.src.SSGetter import SSGetter
from .json_wrapper import load_json, save_json
from .comp_data import get_metadata

"""What follows is the API for interfacing with the Suite-Sparse Matrix 
collection"""

def get_A(mat_name: str) -> scipy.sparse:
    """Get the matrix A 

    Args:
        mat_name (str): Matrix name in the Suite-Sparse collection
    Returns: 
        scipy.sparse: The matrix A in sparse format
    """
    ss_getter = SSGetter()
    return ss_getter.get(mat_name)
    

def load_metadata(
        mat_name: str,
) -> dict:
    """Load the metadata of significance about the given suite-sparse matrix

    Args:
        mat_name (str): the suite sparse matrix

    Returns:
        dict: contains relevant metadata
    """
    A = get_A(mat_name)
    return get_metadata(A)


"""What follows is the API for interfacing with the matrix_meta_data.json"""

def get_matrix_constants(
        matrix_name: str,
) -> tuple[int, float, float, float, float]:
    """Get those constants pertaining to theoretical results, they include:
    NOTE: c, k, and kappa assertain to the power law distribution of the 
    row norms. op-norm is ||A||_2, and n is the number of rows in A.

    Args:
        matrix_name (str): Suite-Sparse matrix name

    Raises:
        ValueError: _description_

    Returns:
        tuple[float, float, float, float, int]: 
            n: number of rows in A
            c: y-intercept of power-law distribution
            k: slope of power-law distribution
            kappa: -3k - 1 (always positive)
            op-norm: ||A||_2
            
    """
    data = load_json(save_path="data/matrix_meta_data.json")

    if matrix_name not in data:
        data[matrix_name] = load_metadata(matrix_name)

    constants = [ 
        "n",
        "c", 
        "k", 
        "kappa", 
        "op-norm",
    ]

    values = [data[matrix_name].get(constant) for constant in constants]

    if None in values:
        raise ValueError(f"One or more constants not found for {matrix_name} in matrix_meta_data.json")
    
    return tuple(values)