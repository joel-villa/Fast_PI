"""
Main Testing suite for svd tests
"""

from scipy.linalg import norm # 2-norm by default
from scipy.sparse.linalg import svds

import numpy as np
from Sparsification_Research.src.Plotter import Plotter
from Sparsification_Research.src.SSGetter import SSGetter
from src.tests.svd import svd_test


def init(mat_name):
    """ Get intial information for tests

    Args: 
        mat_name: the name of the matrix in the SuiteSparse Matrix Collection

    Return:
        A: the matrix in CSR format
    """
    ss_getter = SSGetter(in_csr=False)
    A = ss_getter.get(mat_name)

    return A

if __name__ == '__main__':
    seed = 10
    max_k = 10
    
    mats = [
        "494_bus", # Invalid for mag-based sparsifification
        # "bcsstk07", 
        # "bcsstk08", 
        # "bcsstk19", 
        # "bcsstm07", 
        # "bcspwr06",
        # "impcol_d", # NOT POSITIVE DEFINITE, doesn't work with nystrom sampling
        # "bibd_13_6", # RECTANGULAR: Doesn't work with Nystrom
    ]

    plotter = Plotter(save_fig=False, show_fig=True, fig_size=(12, 6))

    for mat_name in mats:
        A= init(mat_name=mat_name)

        xs, ys, lbl = svd_test(
            A=A, 
            seed=seed,
            max_k=max_k,
            err_type="fro",
        )
        
        plotter.add_to_plot(xs, ys, label=lbl)

        plotter.finish()


