"""A file for getting the metadata of thoeretical significance"""

import scipy

import numpy as np

from collections.abc import Callable

from .json_wrapper import load_json, save_json
from .eig_vect import get_top_eig
from . import comp_data as cd

PATH_M_DATA:str = "data/matrix_meta_data.json"

"""What follows is the API for interfacing with the Suite-Sparse Matrix 
collection"""

def load_and_save_metadata(
        data: dict,
        mat_name: str,
) -> dict:
    """Load the metadata of significance about the given suite-sparse matrix

    Args:
        data (dict): the current dictionary saved under matr_meta_data.json
        mat_name (str): the suite sparse matrix

    Returns:
        dict: The updated dictionary saved in the json file
    """
    matrix_dict = cd.get_metadata(mat_name)

    data = data | {mat_name:matrix_dict}

    save_json(data, save_path=PATH_M_DATA)

    return data

def compute_and_save_metadatum(
        mat_name:str,
        data:dict, 
        data_type:str,        
) -> dict:
    """Compute and save a singular metadatum

    Args:
        mat_name (str): The suitesparse matrix name
        data (dict): The current dictionary stored in the .json
        data_type (str): The type of data missing

    Raises:
        NotImplementedError: Did not implement all metadata options

    Returns:
        dict: the new dictionary stored in the .json
    """
    funct:Callable[[scipy.sparse], float] | None = None

    match data_type:
        # Get only the datum of interest
        case cd.VAR_STRING:
            funct = cd.get_eps_net_var_proxy
        case cd.LAMBDA_STRING:
            # Get only the top eigenvalue of the matrix 'A'
            funct = lambda A: cd.get_lambda_v(A)[0] 
        case _:
            raise NotImplementedError(f"Implement for metadata type: {data_type}")

    A, max_row_mag = cd.preprocess(mat_name)
    datum = funct(A)

    # Update dictionaries
    matrix_dict = data[mat_name]
    matrix_dict = matrix_dict | {data_type:datum} #Add new metadata
    data[mat_name] = matrix_dict

    save_json(data, save_path=PATH_M_DATA)

    return data

"""What follows is the API for interfacing with the matrix_meta_data.json"""

def get_meta_data(
        matrix_name: str,
) -> dict:
    """Get those constants pertaining to theoretical results, they include:
    NOTE: c, k, and kappa assertain to the power law distribution of the 
    row norms. op-norm is ||A||_2, and n is the number of rows in A.

    Args:
        matrix_name (str): Suite-Sparse matrix name

    Raises:
        ValueError: _description_

    Returns:
        dict: A dictionary containing all known meta_data pertaining to the 
        given matrix
    """
    data = load_json(save_path=PATH_M_DATA)

    if matrix_name not in data:
        data = load_and_save_metadata(data, matrix_name)

    matrix_dict = data[matrix_name]
    missing_data = cd.META_DATA - matrix_dict.keys() # What keys are missing?

    for missing_datum in missing_data:
        # compute and save the missing meta datum
        data = compute_and_save_metadatum(matrix_name, data, missing_datum)

    matrix_dict = data[matrix_name]

    return matrix_dict

def get_n_c_k(matrix_name:str) -> tuple[int, float, float]:
    """Get the specified constants

    Args:
        matrix_name (str): the suite-sparse matrix

    Returns:
        tuple[int, float, float]: [n, c, k]
    """
    mat_meta_data = get_meta_data(matrix_name)
    constants = (
        mat_meta_data[cd.N_STRING], 
        mat_meta_data[cd.C_STRING], 
        mat_meta_data[cd.K_STRING],
    )
    return constants

def get_power_consts(
        matrix_name:str
) -> tuple[float, float]:
    """Get the specified constants

    Args:
        matrix_name (str): suite-sparse name

    Returns:
        tuple[float, float]: [c, kappa]
    """
    mat_meta_data = get_meta_data(matrix_name)
    constants = (
        mat_meta_data[cd.C_STRING], 
        mat_meta_data[cd.KAPPA_STRING], 
    )
    return constants

def get_n_norm(
        matrix_name:str,
) -> tuple[int, float]:
    """Get n and the operator norm of the matrix

    Args:
        matrix_name (str): suite sparse matrix name

    Returns:
        tuple[float, float]: [n, op_norm]
    """
    constants = (
        get_meta_data(matrix_name)[cd.N_STRING], 
        get_meta_data(matrix_name)[cd.NORM_STRING],
    )
    return constants

def get_var_proxy(
        matrix_name:str
) -> float:
    """Get the variance proxy of the given matrix

    Args:
        matrix_name (str): SuiteSparse matrix name

    Returns:
        float: ||sum_i(||a_i|| - ||a_i||^2)a_i^Ta_i||
    """
    return get_meta_data(matrix_name)[cd.VAR_STRING]

def get_eig_info(
    mat_name:str,
) -> tuple[float, np.ndarray]:
    """Get the top eigenvalue and eigenvector of the matrix

    Args:
        mat_name (str): SS matrix

    Returns:
        tuple[float, np.ndarray]: [eigenvalue, eigenvector]
    """
    return get_meta_data(mat_name)[cd.LAMBDA_STRING], get_top_eig(mat_name)


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
        "cage7",
        "can_229",
        "dwt_193",
        "eris1176",
        "ex2",
        "fs_541_1",
        "gre_1107",
        "gre_343",
        "hor_131",
        "lshp1561",
        "msc00726",
        "nasa1824",
        "nos3",
        "tomography",
    ]

    mats = sorted(mats) #Alphabetical order

    for mat in mats:
        print(mat)
        val, vect = get_eig_info(mat)
        print(f"val:{val}, vect: {vect[:5]}")
