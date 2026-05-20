"""
Has some runnable methods, which plot the convergence of a JL-enhanced power 
iteration method
"""
from Sparsification_Research.src.SSGetter import SSGetter
from Sparsification_Research.src.Plotter import Plotter

import numpy as np
from scipy.linalg import norm # 2-norm by default
from .tests import power as pwr
from .util import eig_functs as eigs
from .tests import sparse_power as spwr
from .tests import subset_power as sub

def test(funct, plotter, mat_name, seed, num_avg, num_iter, kwargs={}):
    """
    Test the given function from tests.py 
    
    funct   - a fucntion which takes a matrix, some x vals, and a seed, and 
              returns xs and ys to be plotted
    plotter - a Plotter() object
    mat    - a matrix name
    seed    - for randomized reproducability
    num_avg - average of how many tests?
    input   - input for funct
    """
    ss_getter = SSGetter(in_csr=False)
    A = ss_getter.get(mat_name)

    print(mat_name)
    ys = np.zeros(num_iter)
    ys_i = np.zeros(num_iter)

    label = ""

    u_star =  eigs.top_left(A)
    
    for i in range(num_avg):
        seed_i = seed + i * 3 

        rng = np.random.default_rng(seed=seed_i)
        u0 = rng.normal(0,1,np.shape(A)[0])
        u0 = u0 / norm(u0)

        xs, ys_i, label = funct(A, u0, u_star, num_iter, seed=seed_i, **kwargs)
        ys += ys_i
    
    # ys = ys / num_avg

    # """Start of code for sparse_pwr_tol() *** xs & ys are lists of variable 
    # lengths -> averaging complicated and not implemented***"""
    # rng = np.random.default_rng(seed=seed)
    # u0 = rng.normal(0, 1, A.shape[0])
    # u0 = u0 / norm(u0)

    # xs, ys, label = funct(A, u0, u_star, num_iter, seed=seed, **kwargs)
    # END of code for sparse_pwr_tol()

    plotter.add_to_plot(xs, ys, label=label)
    print("Finished test")

def main_swap():
    """
    For testing swap behavior 
    """

    # mats    = ["494_bus"]
    seed    = 10
    num_avg = 1
    num_iter = 64


    # SOME MATS THAT SHOW GOOD BEHVIOR: ["bcsstk07", "bcsstk19", "bcsstm07", "impcol_d"]
    mats = ["bcsstk07", "bcsstk19", "bcsstm07", "impcol_d"]
    mats = ["bcsstk19", "bcsstm07"]
    """These mats seem to imperically have this in common: small spectral gap,
      and large eigenvalues
      TODO: prove why this may be the case? 
      TODO: an itterative approach which will use theses matrix approximations 
            to converge faster"""


    types = ["jl_gaussian", "jl_sparse"]
    ps = [80]
    step_size = 8

    plotter = Plotter(save_fig=False, show_fig=True, fig_size=(12, 6))

    for mat in mats:
        plotter.init_plot(title=f"Power Convergence of {mat}", 
                          x_label="number of iterations",
                          y_label="residual", 
                          save_name=f"{mat}_{ps[0]}_swap",
                          grid_on=True) 

        test(pwr.baseline_pow_convergence, plotter, mat, seed, num_avg, num_iter)
        
        for p in ps:
            for type in types:
                args1 = {"p": p, "type" : type}
                args2 = {"p": p, "step_size": step_size, "type" : type}
                test(pwr.jl_percent_reduced, plotter, mat, seed, num_avg, num_iter, args1)
                test(pwr.multi_jl_p_reduce, plotter, mat, seed, num_avg, num_iter, args2)
            test(sub.percent_subset_pow, plotter, mat, seed, num_avg, num_iter, {"p": p})
            if (mat == "impcol_d"):
                # impcol_d gets "infs" with percent_subset_pow_swap, skipping it for now
                continue 
            test(sub.percent_subset_pow_swap, plotter, mat, seed, num_avg, num_iter, {"p": p, "step_size" : step_size})

        plotter.finish()


def main_no_swap():
    """
    For testing behavior of several approaches inluding:
    - jl reduction
    - column subsets
    - sparsification
    """
    seed    = 10
    num_avg = 5
    num_iter = 64
    p = 97
    step_size = 8
    num_tests = 2


    # SOME MATS THAT SHOW GOOD BEHVIOR: ["bcsstk07", "bcsstk19", "bcsstm07", "impcol_d"]
    mats = ["494_bus", "bcsstk07", "bcsstk08", "bcsstk19", "bcsstm07", "impcol_d"]

    """These mats seem to imperically have this in common: small spectral gap,
      and large eigenvalues
      TODO: prove why this may be the case? 
      TODO: an itterative approach which will use theses matrix approximations 
            to converge faster"""


    types = ["jl_gaussian", "jl_sparse"]

    plotter = Plotter(save_fig=False, show_fig=True, fig_size=(12, 6))

    for mat in mats:
        plotter.init_plot(title=f"Power Convergence of {mat}", 
                          x_label="number of iterations",
                          y_label="residual", 
                          save_name=f"{mat}_subset_no_swap_{p}",
                          grid_on=True) 

        test(pwr.baseline_pow_convergence, plotter, mat, seed, num_avg, num_iter)
        
        for type in types:
            for i in range(num_tests):
                test(pwr.jl_percent_reduced, plotter, mat, seed * i + i, num_avg, num_iter, {"p": p, "type" : type})
            
        for i in range(num_tests):
            test(sub.percent_subset_pow, plotter, mat, seed * i + i, num_avg, num_iter, {"p": p})

        for i in range(num_tests):
            test(funct=spwr.expected_sparse_pwr,
                 plotter=plotter,
                 mat_name=mat,
                 seed=2 + i * 3,
                 num_avg=num_avg,
                 num_iter= num_iter, 
                 kwargs= {"x": 1})

        plotter.finish()


def main_sparsification():
    """
    For testing behavior of sparsification
    """
    seed    = 10
    num_avg = 5
    num_iter = 64

    # Expected percent of sparsification
    ps = [0.001, 0.01, 0.1, 1, 10]
    xs = [1, 4, 16, 64]


    mats = ["494_bus", 
            "bcsstk07", 
            "bcsstk08", 
            "bcsstk19", 
            "bcsstm07", 
            "impcol_d",
            ]

    plotter = Plotter(save_fig=False, show_fig=True, fig_size=(12, 6))


    for mat in mats:
        plotter.init_plot(title=f"Power Convergence of {mat}", 
                          x_label="number of iterations",
                          y_label="residual", 
                          save_name=f"{mat}_sparse_test",
                          grid_on=True) 

        test(funct=pwr.baseline_pow_convergence,
            plotter=plotter, 
            mat_name=mat, 
            seed=seed, 
            num_avg=num_avg, 
            num_iter=num_iter,
            )
        

        tol = None

        # Percentage tests
        for i, p in enumerate(ps):
            test(funct=spwr.percent_sparse_pwr,
                 plotter=plotter,
                 mat_name=mat,
                 seed=2 + i * 3,
                 num_avg=num_avg,
                 num_iter= num_iter, 
                 kwargs= {"p": p, "tol": tol},
                 )
        
        # Expected new zeros tests
        for i, x in enumerate(xs):
            test(funct=spwr.expected_sparse_pwr,
                 plotter=plotter,
                 mat_name=mat,
                 seed=2 + i * 3,
                 num_avg=num_avg,
                 num_iter= num_iter, 
                 kwargs= {"x": x, "tol": tol},
                 )
        plotter.finish()

if __name__ == '__main__':
    # main_no_swap()
    # main_swap()
    main_sparsification()

