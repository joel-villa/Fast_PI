"""A file for getting the metadata of thoeretical significance"""

from .json_wrapper import load_json, save_json
from .comp_data import get_metadata

PATH_M_DATA = "data/matrix_meta_data.json"

THEORY_CONSTANTS = [ 
    "n",
    "c", 
    "k", 
    "kappa", 
    "op-norm",
]

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
    matrix_dict = get_metadata(mat_name)

    data = data | matrix_dict

    save_json(data, save_path=PATH_M_DATA)

    return data

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
    data = load_json(save_path=PATH_M_DATA)

    if matrix_name not in data:
        data[matrix_name] = load_and_save_metadata(data, matrix_name)

    values = [data[matrix_name].get(constant) for constant in THEORY_CONSTANTS]

    if None in values:
        raise ValueError(f"One or more constants not found for {matrix_name} in matrix_meta_data.json")
    
    return tuple(values)


if __name__ == '__main__':
    """Main for testing purposes
    """

