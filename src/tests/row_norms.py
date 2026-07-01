"""
Functions for measuring & displaying row norm distribution

Return:
    xs: [1, 2, ...]
    ys: sorted array of row magnitudes (2-norms)
    lbl: string representation of this test
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

def get_two_norm(mat_name, max_norm):
    """Attempt to read in the matrix data, if none, compute it, and save it

    Args:
        mat_name: the name of the suite-sparse matrix
        max_norm: are we multiplying the matrix by the maximum row-norm?
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

    if max_norm:
        # Assume max_i ||A^(i)|| <= 1
        ys = ys / ys[0]
    

    return xs, ys, mat_name

def get_inner_products(mat_name, max_norm):
    """ Get the inner products of the rows of the matrix (inner product is 
    square of the two-norm)

    Args:
        mat_name: the name of the suite-sparse matrix
        max_norm: are we multiplying the matrix by the maximum row-norm?

    Return:
        xs: [1, 2, ...]
        ys: sorted array of inner products
        lbl: string representation of this test
    """
    xs, ys, lbl = get_two_norm(mat_name=mat_name, max_norm=max_norm)

    ys = np.square(ys)
    return xs, ys, lbl

def get_distribution(mat_name, max_norm, num_bins):
    """ Get the inner products of the rows of the matrix (inner product is 
    square of the two-norm)

    Args:
        mat_name: the name of the suite-sparse matrix
        max_norm: are we multiplying the matrix by the maximum row-norm?
        num_bins: number of bins to use for histogram
    Return:
        ys: The count of each bin
        bin_edges: the edges of the bins 
        lbl: string representation of this test
    """
    xs, ys, lbl = get_two_norm(mat_name=mat_name, max_norm=max_norm)

    return np.histogram(ys, bins=num_bins), lbl

def fit_x_inverse(xs, ys):
    """ fit x and y values to y = m/x + b

    Args:
        xs: x-values
        ys: y-values
    Return:
        xs: x-values (unchanged)
        ys: y-values of a line of best fit
        lbl: string representation
    """
    # Transform x to 1 / x
    x_inverse = 1 / xs

    # Fit a line (degree 1) to the transformed data
    m, b = np.polyfit(x_inverse, ys, deg=1)

    # y = (m/x) + b
    ys = m * (x_inverse) + b
    lbl = f"y = ({m:.4f} / x) + {b:.4f}"

    return xs, ys, lbl

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

def fit_pow_law(xs, ys):
    """ fit x and y values to y = bx^m

    Args:
        xs: x-values
        ys: y-values
    Return:
        xs: x-values (unchanged)
        ys: y-values of a line of best fit
        lbl: string representation
    """
    # Transform x to ln(x), and y to ln(y)
    x_log = np.log(xs)
    y_log = np.log(ys)

    # Fit a line (degree 1) to the transformed data
    m, ln_b = np.polyfit(x_log, y_log, deg=1)

    b = np.exp(ln_b)

    # y = bx^m
    ys = pow_law_y(xs=xs, coefficient=b, exponent=m)
    lbl = fr"$y = {b:,.4f} \cdot x ^{{{m:.4f}}}$"
    return xs, ys, lbl

    # # ln y = m ln x + ln b
    # lbl = fr"$\ln y = m \ln x + \ln {b}$"
    # y_log = m * x_log + ln_b
    # return x_log, y_log, lbl

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

def overfit_pow_law(xs, ys):
    """ Finding a power-law line which overfits the data

    LP Problem:
        [k, ln(a)] = ?
        minimizing [-1, 1][k, ln(a)]^T
        s.t. [ln(x_i), -1][k, ln(a)]^T <= -ln(y_i) - ln(x_i), for all i
        and [k, ln(a)] >= 0

    Args:
        xs: x-values
        ys: y-values
    Return:
        xs: x-values (unchanged)
        ys: y-values of a line of overfit
        lbl: string representation
    """
    # Natural log of data
    ln_xs = np.log(xs)
    ln_ys = np.log(ys)

    # Minimizing [-1, 1][k, ln(a)]^T
    c = np.array([-1, 1])

    # [ln(x_i), -1][k, ln(a)]^T <= -ln(y_i) - ln(x_i), for all i
    A_ub = get_LP_A(ln_xs=ln_xs)
    b_ub = -ln_ys - ln_xs

    bounds = [
        (0 , None),      # k >= 0
        (None, None),   # ln(a) free
    ]

    optimize_result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
    )

    if optimize_result.status != 0:
        # Error occured
        raise ValueError(f"Optimization did not proceed nominally: Error code {optimize_result.status}")

    x = optimize_result.x

    k = x[0]
    ln_a = x[1]

    a = np.exp(ln_a)

    ys = pow_law_y(xs=xs, coefficient=a, exponent= -k - 1)

    print(f"n = {len(xs)}")
    print(fr"$\sum_i ||A^{{(i)}}|| \le {pow_integral(a=a, k= -k - 1, n=len(xs))}$")

    lbl = fr"$y = {a:,.4f} \cdot x ^{{-{k:.4f} - 1}}$"
    
    return xs, ys, lbl
