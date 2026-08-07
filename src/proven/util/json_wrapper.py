"""For reading and writing to json files"""

import json

def load_json(save_path: str) -> dict:
    """Load the data in the json file as a dictionary

    Args:
        save_path (str): path to the json file

    Returns:
        dict: dictionary of the json file
    """
    with open(save_path, 'r') as file:
        data = json.load(file)

    return data

def save_json(data: dict, save_path:str) -> None:
    """Add this data to the dictionary that is currently stored at save_path

    Args:
        data (dict):  dictionary to be added to the json file
        save_path (str): path to the json file
    """
    # Saving to a file
    with open(save_path, 'w') as file:
        json.dump(data, file, indent=2)


"""What follows is the API for interfacing with the matrix_meta_data.json"""
def get_matrix_constants(
        matrix_name: str,
) -> dict:
    """Get those constants pertaining to theoretical results, they include:
    {
        "c": float,
        "k": float,
        "kappa": float,
        "op-norm": float,
        "n": int,
    }
    NOTE: c, k, and kappa assertain to the power law distribution of the 
    row norms. op-norm is ||A||_2, and n is the number of rows in A.

    Args:
        matrix_name (str): Suite-Sparse matrix name

    Returns:
        dict: dictionary of the constants of significance
    """
    data = load_json(save_path="data/matrix_meta_data.json")

    if matrix_name not in data:
        raise ValueError(f"{matrix_name} not found in matrix_meta_data.json")

    constants = ["c", "k", "kappa", "op-norm", "n"]

    values = [data[matrix_name].get(constant) for constant in constants]

    if None in values:
        raise ValueError(f"One or more constants not found for {matrix_name} in matrix_meta_data.json")

    constants = dict(zip(constants, values))

    return constants