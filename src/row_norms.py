"""
Testing the distribution of matrix row norms of Suite-Sparse Matrix collection
"""
import numpy as np

from Sparsification_Research.src.Plotter import Plotter

from .tests import row_norms

if __name__ == '__main__':
    mats = [
        "494_bus", # Invalid for mag-based sparsifification
        "bp_0",
        "bcsstk07", 
        "bcsstk08", 
        "bcsstk19", #TODO: this has odd behavior w/ proven tests (not starting at zero)
        "bcsstm07", 
        "bcspwr06",
        "impcol_d", # NOT POSITIVE DEFINITE, doesn't work with nystrom sampling
        "bibd_13_6", # RECTANGULAR: Doesn't work with Nystrom
    ]

    plotter = Plotter(save_fig=False, show_fig=True, fig_size=(12, 6)) 
    
    for mat_name in mats:
        plotter.init_plot(
            title=f"Two-Norm Distribution ({mat_name})", 
            x_label=r"$\log (\text{row (i)})$",
            y_label=r"$\log||A^{(i)}||$", 
            save_name=f"{mat_name}_two_norm_dist",
            grid_on=True,
        )
        
        xs, ys, lbl = row_norms.get_two_norm(mat_name=mat_name)
        plotter.add_to_plot(xs, ys, lbl)

        #y = m/x + b
        # xs, ys, lbl = row_norms.fit_x_inverse(xs, ys)
        # plotter.add_to_plot(xs, ys, lbl)

        # y = bx^m
        xs, ys, lbl = row_norms.fit_pow_law(xs, ys)
        plotter.add_to_plot(xs, ys, lbl)

        plotter.finish()
        # plotter.finish(xscale='log', yscale='log')