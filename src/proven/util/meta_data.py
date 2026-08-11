"""A file for getting the metadata of thoeretical significance"""

import scipy

from collections.abc import Callable

from .json_wrapper import load_json, save_json
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
        case cd.VAR_STRING:
            funct = cd.get_var_proxy
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

def get_n_c_kappa_norm(matrix_name:str) -> tuple[int, float, float, float]:
    """Get the specified constants

    Args:
        matrix_name (str): suite-sparse name

    Returns:
        tuple[int, float, float, float]: [n, c, kappa, op_norm]
    """
    mat_meta_data = get_meta_data(matrix_name)
    constants = (
        mat_meta_data[cd.N_STRING], 
        mat_meta_data[cd.C_STRING], 
        mat_meta_data[cd.KAPPA_STRING], 
        mat_meta_data[cd.NORM_STRING],
    )
    return constants


if __name__ == '__main__':
    """Main for testing purposes
    """
    mats = [
        # -1.2 > k > -1.1
        "494_bus", # SYMMETRIC

        -1.1 > k > -1.0
        "bcsstk08", #SYMMETRIC
        "ex2", #SYMMETRIC

        # -1.0 > k > -0.9
        "bp_0", #NON SYMMETRIC

        # -0.9 > k > -0.8
        "meg4", #SYMMETRIC
        "1138_bus",


        # -0.8 > k > -0.7
        "hor_131",
        "bcsstk19", #TODO: this has odd behavior w/ proven tests (not starting at zero)
        "nasa1824",
        "bcsstk07", 

        # -0.7 > k > -0.6
        "Harvard500",
        "fs_541_1", 

        # -0.6 > k > -0.5
        "bcsstk34",
        "msc00726",
        "eris1176",

        # -0.5 > k > -0.4
        "qc324", # SYMMETRIC
        "Erdos02",

        # -0.4 > k > -0.3
        "California", 
        "bcsstm07",

        # -0.3 > k > -0.2

        # -0.2 > k > -0.1
        "barth",
        "tomography", 
        "gre_1107",
        "bcspwr10",
        "bcspwr06",
        "gre_1107",
        "dwt_193",
        "gre_343",
        "cage7", 


        # -0.1 > k 
        "blckhole",
        "can_229", 
        "impcol_d", # NOT POSITIVE DEFINITE, doesn't work with nystrom sampling
        "bibd_13_6", # RECTANGULAR: Doesn't work with Nystrom
        "lshp1561",
        "nos3",
        "G1",
        "G67",
    ]

    mats = sorted(mats) #Alphabetical order

    for mat in mats:
        print(f"metadata: {get_meta_data(mat)}")
