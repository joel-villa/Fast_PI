"""For interfacing with the the top eigenvector information of the matrices
stored in top_eig_vect.npz"""

import numpy as np

from comp_data import get_eig_vect
from npz_wrapper import update_npz

PATH_EIG_DATA:str = "data/top_eig_vect.npz"

def get_top_eig(mat:str) -> np.ndarray:
    """Get the top eigenvector of the matrix

    Args:
        mat (str): The SS matrix in question

    Returns:
        np.ndarray: The top eigenvector
    """
    # Read in eigenvector from file (if it exists)
    data_dict = np.load(PATH_EIG_DATA)
    mat_eig_vect = data_dict.get(mat)
    data_dict.close() # Close the file

    # If it doesn't exist compute it, and save it
    if mat_eig_vect is None:
        mat_eig_vect = get_eig_vect(mat)
        update_npz(
            save_path=PATH_EIG_DATA,
            key=mat,
            value=mat_eig_vect,
        )

    return mat_eig_vect