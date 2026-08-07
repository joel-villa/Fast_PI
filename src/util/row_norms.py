"""
Utility functions for row_norm tests
"""

from scipy.optimize import linprog

import json 

import numpy as np
import math

from Sparsification_Research.src.SSGetter import SSGetter
from .constants import THIRTY_TWO_BIT_PRECISION
from ..util.sparse_rows import calc_row_norms

NPZ_PATH = "data/two_norms.npz"
json_path = "data/pow_law.json"

def save_data(mat_name, ys):
    """ Save the matrix's sorted row-norms to data/two_norms.json
    
    Args:
        mat_name: the name of the suite-sparse matrix
        ys: sorted array of row magnitudes (2-norms)
    Return: 
        None
    """
    data = load_data()

    data[mat_name] = ys

    # Save the dictionary by unpacking it with **
    np.savez(NPZ_PATH, **data)

def load_data() -> dict:
    """Read the data 

    Returns:
        dict: _description_
    """
    try:
        with np.load(NPZ_PATH) as loaded_data:
            return dict(loaded_data)
    except EOFError:
        # Empty file error
        return {}
    
def get_norms_from_npz(mat_name:str) -> np.ndarray | None:
    """ Read the matrix's two-norm data from data/two_norms.npz
    
    Args:
        mat_name: 
    Return: 
        None: 
    

    Args:
        mat_name (str): the name of the suite-sparse matrix

    Returns:
        np.ndarray | None: 
            np.ndarray: if two norms are tracked in two_norms.npz
            None: if two norms are not tracked
    """
    data = load_data()

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

def get_sorted_row_norms(mat_name:str, rescale: bool=False) -> np.ndarray:
    """Get the row norms of the given matrix, either from data/two_norms.npz or 
    by calculating them

    Args:
        mat_name (str): the name of the suite-sparse matrix

    Returns:
        np.ndarray: sorted array of row magnitudes (2-norms)
    """
    ys = get_norms_from_npz(mat_name=mat_name)
    
    if ys is None:
        # Matrix data has not been computed, compute them and save
        ys = calculate_two_norm(mat_name=mat_name)
        save_data(mat_name=mat_name, ys=ys)

    if rescale:
        # divide all row magnitudes by the max (i.e.) the top row magnitude is 1
        ys = ys / ys[0]

    return ys

"""THE WORKING STRATEGY FOR POWER LAW USES BRUTE FORCE SEARCH"""

def get_power_law_coefficients(ys: np.ndarray) -> tuple[float, float]:
    """ Get the power law coefficients of the given row-norm distribution 

    Args:
        ys (np.ndarray): the row-magnitudes (sorted)

    Returns:
        tuple[float, float]: 
            float: c or the y-intercept of a log log plot
            float: k or the slope of a log log plot
    """
    num_rows = ys.size
    print(f"ys.size: {num_rows} (assuming this is the number of rows)")
    xs = np.arange(num_rows)

    if np.any(ys < 0):
        raise ValueError("Misuse of overfit_pow_law: y-values should be positive")
    if np.any(ys == 0):
        print("Warning: going to reduce length of xs and ys")
        xs, ys = remove_zero_values(xs=xs, ys=ys)

    # Power law distribution -> ln y = ln c + k ln x
    ln_xs = np.log(xs)
    ln_ys = np.log(ys)
    
    # Testing 1024 values for k in the range -4 to 0
    ks = np.linspace(-4, 0, 1024)

    # Some default settings
    min_area = np.inf
    c_best = np.inf
    k_best = np.inf

    for i, k in enumerate(ks):
        # the y-intercept of a line w/ slope k, that runs above all the data 
        ln_a = np.max(ln_ys - k * ln_xs) 

        c = math.exp(ln_a)
        area = 0

        if abs(k + 1) <= THIRTY_TWO_BIT_PRECISION: #1e-7 ~ 32-bit machine epsilon
            # a ln n, if k == -1
            area = c * math.log(num_rows)
        else: 
            # a / (k+1) (n^(k+1) - 1), if k != -1
            area = c / (k + 1) * (num_rows ** (k + 1) - 1)

        if area < min_area:
            # Found new best k and a settings
            min_area = area
            c_best = c
            k_best = k

    return c_best, k_best


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

def rm_first_funct_vals(xs, ys, function):
    """ Remove the first z values from both xs and ys, where z is the output 
    of function(n), where n is the dimension of xs and ys
    TODO
    Args:
        xs: numpy array of floats
            An array to reduce
        ys: numpy array of floats
            An array to reduce
        function: a function which takes in an array of floats and returns an 
        integer 
            if None, return xs and ys unchanged
    Return: 
        xs: numpy array 
            Subset of original xs
        ys: numpy array
            Subset of original ys
    """
    if function is None:
        return xs, ys
    
    n_subset = function(xs)

    # Only want the last n - n_subset elements
    xs = xs[n_subset:]
    ys = ys[n_subset:]

    return xs, ys
def log_size(xs):
    """ Returns the ceiling of the log of the size of the xs array
    
    Args: 
        xs: numpy array
            Dimension of signifigance
    Return: log(len(xs))
    """
    return math.ceil(math.log(xs.size))