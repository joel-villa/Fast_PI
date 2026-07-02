"""
Functions for measuring & displaying row norm distribution

Return:
    xs: [1, 2, ...]
    ys: sorted array of row magnitudes (2-norms)
    lbl: string representation of this test
"""
from scipy.optimize import linprog

import numpy as np

from ..util import row_norms as util

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
    ys = util.read_data(mat_name=mat_name)

    if ys is None:
        # Matrix data has not been computed, compute them and save
        ys = util.calculate_two_norm(mat_name=mat_name)
        util.save_data(mat_name=mat_name, ys=ys)
    
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

def histogram(mat_name, max_norm, num_bins):
    """ Get the histogram distrubition of the row norms 

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

    bin_counts, bin_edges = np.histogram(ys, bins=num_bins)

    return bin_edges, bin_counts, lbl

def binned_row_weights(mat_name, max_norm, num_bins):
    """ For plotting i vs. iz_i, where i is the bin number and z_i is the 
    number of rows in that bin
    
    Args:
        mat_name: the name of the suite-sparse matrix
        max_norm: are we multiplying the matrix by the maximum row-norm?
        num_bins: number of bins to use for histogram
    Return:
        ys: i * z_i
        xs: i 
        lbl: string representation of this test
    """
    bin_edges, ys, lbl = histogram(mat_name, max_norm, num_bins)

    delta = bin_edges[1] - bin_edges[0]

    xs = np.arange(start=1, stop=len(ys)+1)

    ys = xs * ys * delta

    return xs, ys, r"$i \cdot z_i$"

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
    ys = util.pow_law_y(xs=xs, coefficient=b, exponent=m)
    lbl = fr"$y = {b:,.4f} \cdot x ^{{{m:.4f}}}$"
    return xs, ys, lbl

    # # ln y = m ln x + ln b
    # lbl = fr"$\ln y = m \ln x + \ln {b}$"
    # y_log = m * x_log + ln_b
    # return x_log, y_log, lbl

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
    if np.any(ys < 0):
        raise ValueError("Misuse of overfit_pow_law: y-values should be positive")
    if np.any(ys == 0):
        print("Warning: going to reduce length of xs and ys")
        xs, ys = util.remove_zero_values(xs=xs, ys=ys)

    # Natural log of data
    ln_xs = np.log(xs)
    ln_ys = np.log(ys)

    # Minimizing [-1, 1][k, ln(a)]^T
    c = np.array([-1, 1])

    # FOR x < 1, we want to maxamize k and maximize ln(a): TODO what's up with this? 
    # c = np.array([1, 1])

    # [ln(x_i), -1][k, ln(a)]^T <= -ln(y_i) - ln(x_i), for all i
    A_ub = util.get_LP_A(ln_xs=ln_xs)
    b_ub = -ln_ys - ln_xs

    bounds = [
        (0.0 , None), # k >= 0
        (None, None), # ln(a) free
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

    ys = util.pow_law_y(xs=xs, coefficient=a, exponent= -k - 1)

    print(f"n = {len(xs)}")
    print(fr"$\sum_i ||A^{{(i)}}|| \le {util.pow_integral(a=a, k= -k - 1, n=len(xs))}$")

    lbl = fr"$y = {a:,.4f} \cdot x ^{{-{k:.4f} - 1}}$"
    
    return xs, ys, lbl
