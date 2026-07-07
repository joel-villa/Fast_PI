"""
Functions for measuring & displaying spectral norm distribution of Suitesparse
matrices

Return:
    xs: ...
    ys: ...
    lbl: ...
"""

import json
import numpy as np
import scipy.sparse.linalg as sla

from Sparsification_Research.src.SSGetter import SSGetter

save_path = "data/spect_norms.json"

def load_json(save_path):
    """ Load the json file at save_path
    
    Args: 
        save_path: path to the json file
    
    Return: 
        data: dictionary of the json file
    """
    with open(save_path, 'r') as file:
        data = json.load(file)

    return data

def save_json(data, save_path):
    """ Add this data to the dictionary that is currently stored at save_path

    Args:
        data: dictionary to be added to the json file
        save_path: path to the json file

    Return: 
        None
    """
    # Saving to a file
    with open(save_path, 'w') as file:
        json.dump(data, file, indent=2)

def calculate_2norm(mat_name):
    """ Calculate the spectral norm of the given SS matrix
    
    Args:
        mat_name: string of the SS matrix name
    Return: The spectral norm of the matrix
    """

    ss_getter = SSGetter(in_csr=True)
    A = ss_getter.get(mat_name)

    two_norm = sla.norm(A, ord=2)

    return two_norm

def get_2norms(mats):
    """ Get the spectral norm of the given Suite Sparse matrix
    
    Args:
        mats: list of string of the SS matrix names
    
    Return: A list of the spectral norm of each matrix
    """

    data = load_json(save_path=save_path)

    for mat_name in mats:
        if data.get(mat_name) is None:
            two_norm = calculate_2norm(mat_name)
            data[mat_name] = two_norm
    
    save_json(data, save_path=save_path)

    return  list(data.values())



def histogram(mats, num_bins):
    """ Given a list of matrix names, return the histogram of their spectral 
    norms
    
    Args: 
        mats: matrix name list
        num_bins: number of bins for the histogram
    
    Return:
        xs: bin edges for the histogram 
        ys: histogram values  
        lbl: Title of the histogram
    """

    spect_norms = get_2norms(mats)

    bin_counts, bin_edges = np.histogram(spect_norms, bins=num_bins)

    return bin_edges, bin_counts, f"Spectral Norm Histogram of {len(mats)} SS Matrices"