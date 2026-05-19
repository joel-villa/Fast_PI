"""
Has some runnable methods, which plot the convergence of a JL-enhanced power 
iteration method
"""
from Sparsification_Research.src.SSGetter import SSGetter
from Sparsification_Research.src.Plotter import Plotter

import numpy as np
from .tests.power import *
from .util.eig_functs import *
from .tests.sparse_power import sparse_pow
from .tests.subset_power import percent_subset_pow
from .tests.subset_power import percent_subset_pow_swap

def test(funct, plotter, mat_name, seed, num_avg, num_iter, args={}):
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

    u_star =  top_left(A)
    
    for i in range(num_avg):
        seed_i = seed + i 

        rng = np.random.default_rng(seed=seed_i)
        u0 = rng.normal(0,1,np.shape(A)[0])
        u0 = u0 / norm(u0)

        xs, ys_i, label = funct(A, u0, u_star, num_iter, seed=seed_i, **args)
        ys += ys_i
    
    ys = ys / num_avg

    plotter.add_to_plot(xs, ys, label=label)
    print("Finished test")

def main_swap():
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

    plotter = Plotter(save_fig=True, show_fig=True, fig_size=(12, 6))

    for mat in mats:
        plotter.init_plot(title=f"Power Convergence of {mat}", 
                          x_label="number of iterations",
                          y_label="residual", 
                          save_name=f"{mat}_{ps[0]}_swap",
                          grid_on=True) 

        test(baseline_pow_convergence, plotter, mat, seed, num_avg, num_iter)
        
        for p in ps:
            for type in types:
                args1 = {"p": p, "type" : type}
                args2 = {"p": p, "step_size": step_size, "type" : type}
                test(jl_percent_reduced, plotter, mat, seed, num_avg, num_iter, args1)
                test(multi_jl_p_reduce, plotter, mat, seed, num_avg, num_iter, args2)
            test(percent_subset_pow, plotter, mat, seed, num_avg, num_iter, {"p": p})
            if (mat == "impcol_d"):
                # impcol_d gets "infs" with percent_subset_pow_swap, skipping it for now
                continue 
            test(percent_subset_pow_swap, plotter, mat, seed, num_avg, num_iter, {"p": p, "step_size" : step_size})

        plotter.finish()


def main_no_swap():
    # mats    = ["494_bus"]
    seed    = 10
    num_avg = 1
    num_iter = 64


    # SOME MATS THAT SHOW GOOD BEHVIOR: ["bcsstk07", "bcsstk19", "bcsstm07", "impcol_d"]
    mats = ["bcsstk07", "bcsstk19", "bcsstm07", "impcol_d"]
    mats = ["bcsstk19"]
    """These mats seem to imperically have this in common: small spectral gap,
      and large eigenvalues
      TODO: prove why this may be the case? 
      TODO: an itterative approach which will use theses matrix approximations 
            to converge faster"""


    types = ["jl_gaussian", "jl_sparse"]
    types = ["jl_gaussian"]
    p = 97
    step_size = 8

    plotter = Plotter(save_fig=False, show_fig=True, fig_size=(12, 6))

    for mat in mats:
        plotter.init_plot(title=f"Power Convergence of {mat}", 
                          x_label="number of iterations",
                          y_label="residual", 
                          save_name=f"{mat}_subset_no_swap_{p}",
                          grid_on=True) 

        test(baseline_pow_convergence, plotter, mat, seed, num_avg, num_iter)
        
        for type in types:
            for i in range(7):
                test(jl_percent_reduced, plotter, mat, seed * i + i, num_avg, num_iter, {"p": p, "type" : type})
            
        # for i in range(7):
        #     test(percent_subset_pow, plotter, mat, seed * i + i, num_avg, num_iter, {"p": p})

        plotter.finish()

if __name__ == '__main__':
    main_no_swap()

