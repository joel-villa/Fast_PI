"""
Testing the distribution of matrix row norms of Suite-Sparse Matrix collection
"""
import numpy as np
import matplotlib.pyplot as plt

from Sparsification_Research.src.Plotter import Plotter

from .tests import row_norms
from .util.row_norms import remove_zero_values, rm_first_funct_vals, log_size

def plot(plotter, xs, ys, lbl, loglog):
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

def histogram(mats, loglog):
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

def histogramish(mats, loglog):
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
def row_norms_dist(mats, loglog, funct):
    """ Plot row vs. row-norms for matrices
    
    Args:
        mats: list of matrix names
        loglog: boolean, whether to use log-log scale
        funct:  removing the first funct(x) values from the dataset
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
        "barth",
        
        #~x^-0.75
        "bcsstk08", #SYMMETRIC

        #~x^-0.3
        "Erdos02",
        "Harvard500", 

        #0.25
        "California", 

        #0.2
        "qc324", # SYMMETRIC
        "494_bus", # Invalid for mag-based sparsifification

        # x^.1
        "nasa1824",
        "bp_0",
        "tomography", 
        "gre_1107",
        "ex2",


        #x^0.05
        "bcspwr10",
        "gre_343",
        "hor_131",
        "bcspwr06",
        "can_229", 
        "gre_1107",

        #x^0
        "blckhole",
        "bcsstm07",
        "bcsstk19", #TODO: this has odd behavior w/ proven tests (not starting at zero)
        "fs_541_1",
        "bcsstk07", 
        "cage7", 
        "dwt_193",
        "impcol_d", # NOT POSITIVE DEFINITE, doesn't work with nystrom sampling
        "bibd_13_6", # RECTANGULAR: Doesn't work with Nystrom
        "eris1176",
        "lshp1561",
        "nos3",
        "bcsstk34",
        "msc00726",
        "meg4",
        "G1",
        "G67",

    ]
    
    loglog = True
    subset = log_size

    row_norms_dist(mats, loglog, subset)
    # histogram(mats, loglog)
    # histogramish(mats, loglog)