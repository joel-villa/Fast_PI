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

def save_data(mat_name, ys):
    """ Save the matrix's sorted row-norms to data/two_norms.json
    
    Args:
        mat_name: the name of the suite-sparse matrix
        ys: sorted array of row magnitudes (2-norms)
    Return: 
        None
    """

def read_data(mat_name):
    """ Read the matrix's two-norm data from data/two_norms.json
    
    Args:
        mat_name: the name of the suite-sparse matrix
    Return: 
        None: if DNE in two_norms.json, ys otherwise
    """
    with open('data.json') as json_file:
        data = json.load(json_file)

    # Dictionary lookup, w/ default of None
    ys = data.get(mat_name)

    if ys is None:
        # Matrix data has not been computed, compute them
        return calculate_two_norm(mat_name=mat_name)
    else:
        return ys

def calculate_two_norm(mat_name):
    """ Data not stored in data/two_norm.json, must be calculated

    Args:
        mat_name: the name of the suite-sparse matrix
    Return:
        ys: sorted array of row magnitudes (2-norms)
    """

def get_two_norm(mat_name):
    """Attempt to read in the matrix data, if none, compute it, and save it

    Args:
        mat_name: the name of the suite-sparse matrix
    Return:
        xs: [1, 2, ...]
        ys: sorted array of row magnitudes (2-norms)
        lbl: string representation of this test
    """