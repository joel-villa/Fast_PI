"""
Testing the distribution of matrix row norms of Suite-Sparse Matrix collection
"""
from collections.abc import Callable

import numpy as np
import matplotlib.pyplot as plt

from Sparsification_Research.src.Plotter import Plotter

from .tests import row_norms
from .util.row_norms import remove_zero_values, rm_first_funct_vals, log_size

def plot(
        plotter: Plotter, 
        xs: np.ndarray, 
        ys: np.ndarray, 
        lbl: str, 
        loglog: bool,
    ) -> None:
    """ Add the xs, ys, lbl to the plot, if loglog, take log of xs and ys
    
    Args:
        plotter: Plotter object
        xs: x-values of the plot
        ys: y-values of the plot
        lbl: string representation of this test
        loglog: boolean, whether to use log-log scale

    Return: NONE
    """
    if loglog:
        if np.any(ys <= 0):
            # don't log non-positive
            xs, ys = remove_zero_values(xs, ys)
        plotter.add_to_plot(np.log(xs), np.log(ys), lbl)
    else: 
        plotter.add_to_plot(xs, ys, lbl)

def histogram(mats:list[str], loglog:bool) -> None:
    """ Plot the histogram of row norms for a given matrix
    
    Args:
        mats: list of matrix names
        loglog: boolean, whether to use log-log scale
    Return: NONE
    """

    for mat_name in mats:
        # Histogram
        bin_edges, bin_counts, lbl = row_norms.histogram(mat_name=mat_name, max_norm=False, num_bins=100)

        # Stairs -> histogram
        if loglog:
            plt.stairs(np.log(bin_counts), np.log(bin_edges), fill=True, color='skyblue', edgecolor='black')
        else: 
            plt.stairs(bin_counts, bin_edges, fill=True, color='skyblue', edgecolor='black')

        plt.title(f"Two-Norm Distribution ({mat_name})")
        
        if loglog:
            plt.xlabel("log(Row Magnitude (2-Norm))")
            plt.ylabel("log(Number of Rows)")
        else: 
            plt.xlabel("Row Magnitude (2-Norm)")
            plt.ylabel("Number of Rows")
        plt.show()

def histogramish(mats:list[str], loglog:bool) -> None:
    """ Plot something akin to the histogram of row norms for a given matrix
    
    Args:
        mats: list of matrix names
        loglog: boolean, whether to use log-log scale
    Return: NONE
    """
    plotter = Plotter(save_fig=False, show_fig=True, fig_size=(12, 6)) 

    x_label=r"$i$ (Bin Number)"
    y_label=r"Bin Weight ($i \cdot z_i$)"

    if loglog:
        x_label=r"$\log i$ (Bin Number)"
        y_label=r"Bin Weight ($\log {i \cdot z_i}$)"

    for mat_name in mats:
        plotter.init_plot(
            title=f"Two-Norm Distribution ({mat_name})", 
            x_label=x_label,
            y_label=y_label, 
            save_name=f"{mat_name}_two_norm_dist",
            grid_on=True,
        )

        # Histogram(ish)
        xs, ys_star, lbl = row_norms.binned_row_weights(mat_name=mat_name, max_norm=True, num_bins=100)
        print(f"xs = {xs[:10]}")
        print(f"ys_star = {ys_star[:10]}")
        plot(plotter, xs, ys_star, lbl, loglog)


        # y <= ax^k
        # xs, ys, lbl = row_norms.overfit_pow_law_v3(xs, ys_star)
        # plot(plotter, xs, ys, lbl, loglog)

        
        plotter.finish()
def row_norms_dist(
        mats: list[str], 
        loglog: bool, 
        funct: Callable[[np.ndarray], int], 
        f_of_y: Callable[[np.ndarray], np.ndarray],
    ) -> None:
    """ Plot row vs. row-norms for matrices
    
    Args:
        mats: list of matrix names
        loglog: boolean, whether to use log-log scale
        funct:  removing the first funct(x) values from the dataset
        f_of_y: a function to apply to the y-values before plotting & fitting
    Return: NONE
    """
    plotter = Plotter(save_fig=False, show_fig=True, fig_size=(12, 6)) 

    x_label=r"row ($i$)"
    y_label=r"$||A^{(i)}||$" 

    if loglog:
        x_label=r"$\log (\text{row (i)})$"
        y_label=r"$\log||A^{(i)}||$"    
         
    for mat_name in mats:
        plotter.init_plot(
            title=f"Two-Norm Distribution ({mat_name})", 
            x_label=x_label,
            y_label=y_label, 
            save_name=f"{mat_name}_two_norm_dist",
            grid_on=True,
        )

        # xs, ys, lbl = row_norms.get_two_norm(mat_name=mat_name)
        xs, ys_star, lbl = row_norms.get_two_norm(mat_name=mat_name, max_norm=True)
        ys_star = f_of_y(ys_star)

        #RESORT
        ys_star = np.sort(ys_star)[::-1] #[::-1] -> descending

        plot(plotter, xs, ys_star, lbl, loglog)

        #y = m/x + b
        # xs, ys, lbl = row_norms.fit_x_inverse(xs, ys)
        # xs, ys = rm_first_funct_vals(xs, ys_star, funct)
        # plotter.add_to_plot(xs, ys, lbl)

        # y = bx^m
        # xs, ys, lbl = row_norms.fit_pow_law(xs, ys_star)
        #xs, ys = rm_first_funct_vals(xs, ys_star, funct)
        # plot(plotter, xs, ys, lbl, loglog)

        # y <= ax^k
        x_subset, y_star_subset = rm_first_funct_vals(xs, ys_star, funct)
        # print(f"x_sub = {x_subset[:5]}, y_sub = {y_star_subset[:5]}")
        xs, ys, lbl = row_norms.overfit_pow_law_v3(x_subset, y_star_subset)
        plot(plotter, xs, ys, lbl, loglog)

        plotter.finish()

if __name__ == '__main__':
    mats = [

        # -1.2 > k > -1.1
        "494_bus", # SYMMETRIC

        # -1.1 > k > -1.0
        "bcsstk08", #SYMMETRIC
        "ex2", #SYMMETRIC

        # -1.0 > k > -0.9
        "bp_0", #NON SYMMETRIC

        # -0.9 > k > -0.8
        "meg4", #SYMMETRIC
        "1138_bus",


        # -0.8 > k > -0.7
        "hor_131",
        "bcsstk19", #TODO: this has odd behavior w/ proven tests (not starting at zero)
        "nasa1824",
        "bcsstk07", 

        # -0.7 > k > -0.6
        "Harvard500",
        "fs_541_1", 

        # -0.6 > k > -0.5
        "bcsstk34",
        "msc00726",
        "eris1176",

        # -0.5 > k > -0.4
        "qc324", # SYMMETRIC
        "Erdos02",

        # -0.4 > k > -0.3
        "California", 
        "bcsstm07",

        # -0.3 > k > -0.2

        # -0.2 > k > -0.1
        "barth",
        "tomography", 
        "gre_1107",
        "bcspwr10",
        "bcspwr06",
        "gre_1107",
        "dwt_193",
        "gre_343",
        "cage7", 


        # -0.1 > k 
        "blckhole",
        "can_229", 
        "impcol_d", # NOT POSITIVE DEFINITE, doesn't work with nystrom sampling
        "bibd_13_6", # RECTANGULAR: Doesn't work with Nystrom
        "lshp1561",
        "nos3",
        "G1",
        "G67",
    ]
    
    loglog = True
    subset = None
    # subset = log_size
    f_of_y = lambda y: y ** 3 - y ** 4

    row_norms_dist(mats, loglog, subset, f_of_y=f_of_y)
    # histogram(mats, loglog)
    # histogramish(mats, loglog)