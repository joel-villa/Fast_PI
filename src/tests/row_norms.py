"""
Functions for measuring & displaying row norm distribution

Args:
    mat_name: the name of the suite-sparse matrix
Return:
    xs: [1, 2, ...]
    ys: sorted array of row magnitudes (2-norms)
    lbl: string representation of this test
"""

import json 

import numpy as np

from Sparsification_Research.src.SSGetter import SSGetter
from ..util.sparse_rows import calc_row_norms

save_path = "data/two_norms.json"

def save_data(mat_name, ys):
    """ Save the matrix's sorted row-norms to data/two_norms.json
    
    Args:
        mat_name: the name of the suite-sparse matrix
        ys: sorted array of row magnitudes (2-norms)
    Return: 
        None
    """
    data = read_data()

    data[mat_name] = ys.tolist()

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)

def read_data(mat_name=None):
    """ Read the matrix's two-norm data from data/two_norms.json
    
    Args:
        mat_name: the name of the suite-sparse matrix
    Return: 
        None: if DNE in two_norms.json, ys otherwise
    """
    with open(save_path, 'r') as file:
        data = json.load(file)

    if mat_name is None:
        return data

    # Dictionary lookup, w/ default of None
    return data.get(mat_name)

def calculate_two_norm(mat_name):
    """ Data not stored in data/two_norm.json, must be calculated

    Args:
        mat_name: the name of the suite-sparse matrix
    Return:
        ys: sorted array of row magnitudes (2-norms)
    """

    ss_getter = SSGetter(in_csr=True)
    A = ss_getter.get(mat_name)
    
    row_norms = calc_row_norms(A=A, ord=2) # The 2-norm of each row

    row_norms = np.sort(row_norms) # sorted
    
    row_norms = row_norms[::-1] #descending

    return row_norms

def get_two_norm(mat_name):
    """Attempt to read in the matrix data, if none, compute it, and save it

    Args:
        mat_name: the name of the suite-sparse matrix
    Return:
        xs: [1, 2, ...]
        ys: sorted array of row magnitudes (2-norms)
        lbl: string representation of this test
    """
    ys = read_data(mat_name=mat_name)

    if ys is None:
        # Matrix data has not been computed, compute them and save
        ys = calculate_two_norm(mat_name=mat_name)
        save_data(mat_name=mat_name, ys=ys)
    
    rows = ys.shape[0] # The number of rows

    xs = range(rows)

    return xs, ys, mat_name