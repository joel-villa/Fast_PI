"""For interfacing directly with npz files"""

import numpy as np

def read_npz(save_path:str) -> dict:
    """Read the dictionary inside the npz file

    Args:
        save_path (str): The path to the npz file

    Returns:
        dict: The dictionary saved in the npz file
    """
    try:
        dictionary = dict(np.load(save_path))
    except:
        # Only occurs on first instantiation i.e. nothing exists in the file
        dictionary = {}
    
    return dictionary

def save_npz(save_path:str, data: dict) -> None:
    """Add this data to the dictionary that is currently stored at save_path

    Args:
        save_path (str): path to the npz file
        data (dict):  dictionary to be added to the npz file
    """
    np.savez(save_path, **data)

def update_npz(save_path:str, key:str, value:np.ndarray) -> None:
    """Add the following information to the npz file

    Args:
        save_path: path to the npz file
        key (str): The matrix name
        value (np.ndarray): The top eigenvector of said matrix
    """
    dict = read_npz(save_path=save_path)
    dict = dict | {key:value} #Add key,value pair to the dictionary
    save_npz(save_path, dict)