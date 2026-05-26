"""
For plotting tests which take into account work required to get a 
top-eigenvector approximation, with some score

Where work is number of scalar mults and "score" is closeness of singular value
"""

from wsgiref import types

from scipy.linalg import norm # 2-norm by default
from scipy.sparse.linalg import svds

import numpy as np
from Sparsification_Research.src.Plotter import Plotter
from Sparsification_Research.src.SSGetter import SSGetter

from src.tests import score as tst
from src.tests import score_sample as smpl
from src.tests.power_lazy import lazy_percent

def test(funct, plotter, num_avg, A, num_iter, kwargs={}):
    """Test the given function from src/tests
    
    Args:
        funct: a fucntion which takes a matrix, some x vals, and a seed, and 
               returns xs and ys to be plotted
        plotter: a Plotter() object
        num_avg: average of how many tests?
        A: a CSR sparse matrix
        num_iter: how many iterations?
        seed: for randomized reproducability
        kwargs: input for funct
    Return: 
        The number of iterations ran (important for tests which terminate after
        converged)
    """
    ys = np.zeros(num_iter)
    xs = np.zeros(num_iter)
    ys_i = np.zeros((num_avg, num_iter)) # for averaging

    label = ""

    max_iter = 0
    min_iter = num_iter
    
    for i in range(num_avg):
        if "seed" in kwargs:
            # If seed in kwargs, update it for variable tests to average
            kwargs["seed"] = kwargs["seed"] * (i + 1) + i * 3

        xs_temp, ys_i[i], label = funct(A=A, max_iter=num_iter, **kwargs)

        trim_size = np.trim_zeros(xs_temp).shape[0]

        if (min_iter > trim_size):
            # Smaller x value, store this one for consistent averaging
            xs = xs_temp
            min_iter = trim_size

    # Calculate average
    ys = np.mean(ys_i, axis=0)

    # Remove trailing zeros only ('b' for back)
    xs = np.trim_zeros(xs, trim='b')
    ys = np.trim_zeros(ys, trim='b')

    if (ys.shape[0] < xs.shape[0]):
        # print(f"ys: [{ys[:3]}, ..., {ys[-3:]}], xs: [{xs[:3]}, ..., {xs[-3:]}]")
        raise ValueError("ys has fewer non-zero entries than xs, cannot plot")
    
    ys = ys[:xs.shape[0]]

    #UNCOMMENT FOLLOWING LINE FOR ITTERATIVE VIEW
    # xs = np.arange(xs.shape[0])

    """For plotting initial values (since using log-scale-x, need to add 
    perturbation to see x=0 values)"""
    # xs = xs + 1e-2

    plotter.add_to_plot(xs, ys, label=label)

def init(mat_name, seed, tol):
    """ Get intial information for tests

    Args: 
        mat_name: the name of the matrix in the SuiteSparse Matrix Collection
        seed: for generating initial guess for top left eigenvector

    Return:
        A: the matrix in CSR format
        kwargs: a dictionary containing the initial guess for u, as wall as the 
                solution u
    """
    ss_getter = SSGetter(in_csr=False)
    A = ss_getter.get(mat_name)
        
    _, s_star, _ =  svds(A, k=1)

    s_star = s_star[0]

    rng = np.random.default_rng(seed=seed)
    u0 = rng.normal(0, 1, A.shape[0])
    u0 = u0 / norm(u0)

    print(f"Testing {mat_name}")

    kwargs = {"s_star": s_star,
              "u_0": u0,
              "tol": tol,
              }

    return A, kwargs

def jl_reduction(funct_args, kwargs):
    """Test the naive JL enhancement to power iteration
    
    Args:
        funct_args: a dictionary of arguments for the function jl_percent()
                    (except for type, and p)
        kwargs: a dictionary of arguments for the test function
    
    Return: None
    """
    ps = (80, 97)
    types = ("simple", "gaussian", "sparse")

    for p in ps:
        for type in types:
            p_args = funct_args | {"type": type, "p": p}
            test(funct=tst.jl_percent,
                 kwargs=p_args,
                 **kwargs)

def jl_lazy(funct_args, ps, kwargs):
    """Test the JL enhancement to power iteration, with lazy multiplication
    
    Args:
        funct_args: dictionary of arguments for the function lazy_percent() 
                    (except for p)
        ps: the percent reduction ammounts
        kwargs: a dictionary of arguments for the test function
        
    Return: None
    """
    for p in ps:
        p_args = funct_args | {"p": p}
        test(funct=lazy_percent,
            kwargs=p_args,
            **kwargs)
        
def col_sample(ps, funct_args, kwargs, types):
    """Test column sampling enhancement to power iteration

    Args: 
        ps: the percent reduction ammounts
        funct_args: dictionary of arguments for the function col_p_sample() 
                    (except for p and type)
        kwargs: a dictionary of arguments for the test function
        types: a list of string representations of the types of column sampling

    Return: None
    """
    for p in ps: 
        for type in types:
            p_args = funct_args | {"p": p, "type": type}
            test(funct=smpl.col_sample_p,
                 kwargs=p_args,
                 **kwargs)
            
def sparsification(funct_args, kwargs):
    """Test sparsification enhancement to power iteration
    
    Args: 
        funct_args: a dictionary of arguments for the function sparsification() 
                    (except for p and type)
        kwargs: a dictionary of arguments for the test function
    
    Return: None
    """
    ps = (2, 16)
    types = ("MD", "Generic")

    for p in ps:
        for type in types:
            p_args = funct_args | {"p": p, "type": type}
            test(
                funct=tst.percent_sparse,
                kwargs=p_args,
                **kwargs,
            )

def col_sample_dec(funct_args, kwargs, types):
    """Test column sampling enhancement to power iteration

    Args: 
        funct_args: dictionary of arguments for the function col_p_sample() 
                    (except for p and type)
        kwargs: a dictionary of arguments for the test function
        types: a list of string representations of the types of column sampling

    Return: None
    """
    init_p = 98
    step = 2
    dec_funct = lambda p: p - 5 #TODO: debug this

    funct_args = funct_args | {
        "p0": init_p, 
        # "step": step, 
        "dec_funct": dec_funct,
        "swap_tol": 0.1, # override tol for this test (diff-behavior)
    }

    for type in types:
        p_args = funct_args | {"type": type}
        test(funct=smpl.col_sample_dec_p,
             kwargs=p_args,
             **kwargs)

if __name__ == '__main__':
    seed = 10
    max_iter = 64
    num_avg = 1
    ps = (25, 50, 97)
    tol = 1e-7
    epsilon = 0.98
    sample_types = ("simple", "1-norm")
    
    # SOME MATS THAT SHOW GOOD BEHVIOR: ["bcsstk07", "bcsstk19", "bcsstm07", "impcol_d"]
    mats = ["494_bus", "bcsstk07", "bcsstk08", "bcsstk19", "bcsstm07", "impcol_d", "bcspwr06"]
    # mats = ["bcspwr06"]

    plotter = Plotter(save_fig=False, show_fig=True, fig_size=(12, 6))

    for mat_name in mats:
        A, funct_args = init(mat_name=mat_name, seed=seed, tol=tol)

        plotter.init_plot(title=f"Power Convergence of {mat_name}", 
                          x_label="work (approximate number of scalar multiplications)",
                          y_label=r"Error Rate $\left(\frac{|\lambda^* - \lambda|}{|\lambda^*|}\right)$", 
                          save_name=f"{mat_name}_error_v_work",
                          grid_on=True) 
        
        kwargs = {"plotter": plotter,
                  "num_iter": max_iter,
                  "A": A,
                  }
        
        # Baseline test
        test(funct=tst.baseline,
             num_avg=1,
             kwargs=funct_args,
             **kwargs,
             )
        
        # The following tests have some randomness, so we average them over num_avg runs
        kwargs = kwargs | {"num_avg" : num_avg} 
        # Add seed to funct_args for tests which require randomness
        funct_args = funct_args | {"seed": seed}

        # Args for JL reduction tests
        jl_args = funct_args | {"eps": epsilon}

        # jl_reduction(funct_args=jl_args, kwargs=kwargs)
        # jl_lazy(funct_args=jl_args, ps=ps, kwargs=kwargs)

        col_sample(
            ps=ps, 
            funct_args=funct_args, 
            kwargs=kwargs, 
            types=sample_types
        )

        # sparsification(funct_args=funct_args, kwargs=kwargs)

        col_sample_dec(
            funct_args=funct_args, 
            kwargs=kwargs, 
            types=sample_types
        )

        plotter.finish(
            # xscale="log"
            )