"""
Functions for measuring & displaying row norm distribution

Return:
    xs: [1, 2, ...]
    ys: sorted array of row magnitudes (2-norms)
    lbl: string representation of this test
"""

import json 

import numpy as np

from Sparsification_Research.src.SSGetter import SSGetter
from ..util.sparse_rows import calc_row_norms

save_path = "data/two_norms.npz"
json_path = "data/pow_law.json"

def save_data(mat_name, ys):
    """ Save the matrix's sorted row-norms to data/two_norms.json
    
    Args:
        mat_name: the name of the suite-sparse matrix
        ys: sorted array of row magnitudes (2-norms)
    Return: 
        None
    """
    data = read_data()

    data[mat_name] = ys

    # Save the dictionary by unpacking it with **
    np.savez(save_path, **data)
    
def read_data(mat_name=None):
    """ Read the matrix's two-norm data from data/two_norms.json
    
    Args:
        mat_name: the name of the suite-sparse matrix
    Return: 
        None: if DNE in two_norms.json, ys otherwise
    """
    try:
        with np.load(save_path) as loaded_data:
            data = dict(loaded_data)
    except EOFError:
        # Empty file error
        if mat_name is None:
            return {}
        return None 

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
    print("SHOULDN'T BE PRINTING")

    ss_getter = SSGetter(in_csr=True)
    A = ss_getter.get(mat_name)
    
    row_norms = calc_row_norms(A=A, ord=2) # The 2-norm of each row

    row_norms = np.sort(row_norms) # sorted
    
    row_norms = row_norms[::-1] #descending

    return row_norms

def pow_law_calcs(mat_name, ys):
    num_rows = ys.shape[0]

    twenty_percent_rows = num_rows // 5 # Floor division

    # The sum of the two-norm of the top 20% of rows
    top_20 = ys[:twenty_percent_rows]
    print(f"top_20.shape = {top_20.shape}")
    top_20_sum = np.sum(top_20) 

    # The sum of the two-norm of the bottom 20% of rows
    bottom_80 = ys[twenty_percent_rows:]
    print(f"bottom_80.shape = {bottom_80.shape}")
    bottom_80_sum = np.sum(bottom_80)

    # Reading from a file
    with open(json_path, 'r') as file:
        data = json.load(file)

    data[mat_name] = {
        "top    20% sum" : top_20_sum,
        "bottom 80% sum" : bottom_80_sum,

    }

    # Saving to a file
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=2)

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

    xs = np.arange(start=1, stop=rows+1)

    pow_law_calcs(mat_name=mat_name, ys=ys)

    return xs, ys, mat_name

def fit_x_inverse(xs, ys):
    # Transform x to 1 / x
    x_inverse = 1 / xs

    # Fit a line (degree 1) to the transformed data
    m, b = np.polyfit(x_inverse, ys, deg=1)

    # y = (m/x) + b
    ys = m * (x_inverse) + b
    lbl = f"y = ({m:.4f} / x) + {b:.4f}"

    return xs, ys, lbl