"""For tracking info about row magnitudes of the matrices (UNSORTED)"""

import numpy as np

from .npz_wrapper import update_npz, read_npz

from .comp_data import comp_row_mags

PATH_ROW_MAGS:str = "data/row_mags_unsorted.npz"

def get_row_mags(mat:str) -> np.ndarray:
    """Get the row magniuteds of the matrices

    Args:
        mat (str): The SS matrix in question

    Returns:
        np.ndarray: The row magnitudes of the matrix in order
    """
    # Read in eigenvector from file (if it exists)
    data_dict = read_npz(PATH_ROW_MAGS)
    mat_row_mags = data_dict.get(mat)

    # If it doesn't exist compute it, and save it
    if mat_row_mags is None:
        mat_row_mags = comp_row_mags(mat)
        update_npz(
            save_path=PATH_ROW_MAGS,
            key=mat,
            value=mat_row_mags,
        )

    return mat_row_mags