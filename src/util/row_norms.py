"""
Utility functions for row_norm tests
"""

from scipy.optimize import linprog

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

""" 
POWER LAW RELATED FUNCTIONS BELOW
"""

def pow_law_y(xs, coefficient, exponent):
    """ Ge the power law values of x 

    ys = a * x^k
    
    Args:
        xs: x-values
        coefficient: mutlitipicative constant of power law
        exponent: exponent of power law
    Return:
        ys: array of power law values
    """
    return  coefficient * (xs ** exponent)

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

def get_LP_A(ln_xs):
    """Get the A matrix used in the linear programming problem
    
    Args: 
        ln_xs: natural log of those x-values
    Return: 
        A_ub: can be plugged directly into scipy.optimize.linprog
    """
    # The column vectors
    xT = ln_xs.reshape(-1, 1)
    neg_ones = -1 * np.ones_like(xT)

    # The matrix
    A_ub = np.hstack((xT, neg_ones))
    return A_ub

def pow_integral(a, k, n):
    """ The integral of ax^k, from x=1 to n
    
    Args: 
        a: coefficient
        k: power
        n: number of rows in A
    Return:
        a / (k + 1) * (n^(k + 1) - 1), if k is not -1
        a * ln(n), if k is -1
    """

    if k == -1:
        return a * np.log(n)
    else: 
        return a / (k + 1) * (n ** (k + 1) - 1)
    
def remove_zero_values(xs, ys):
    """ Remove zero values from ys, and corresponding values from xs

    Args:
        xs: x-values
        ys: y-values
    Return:
        xs: x-values w/o zero values
        ys: y-values w/o corresponding values 
    """
    # Remove zero values from xs and ys
    nonzero_indices = np.nonzero(ys)[0]
    xs = xs[nonzero_indices]
    ys = ys[nonzero_indices]
    return xs, ys