from Sparsification_Research.src.MDSparsifier import MDSparsifier
from Sparsification_Research.src.SGenerator import SGenerator

import numpy as np

from ..util.eig_functs import euclidean_dist
from ..util import power as pwr

def sparse_pwr(A, u_0, u_star, num_iter, seed, s):
    sparsifier = MDSparsifier(seed=seed)

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)


    sparse_A = A.copy()

    sparsifier.sparsify(sparse_A, s)

    v =  pwr.v_from_u(sparse_A, u_0)

    # Initial residual
    xs[0] = 0
    ys[0] = euclidean_dist(u_0, u_star)

    
    for i in range(1, num_iter):
        u, _, v = pwr.topsing(v0=v,
                          A=sparse_A, 
                          maxiter=1)

        # Track x and y
        xs[i] = i
        ys[i] = euclidean_dist(u, u_star)
    
    # print(ys)
    num_mults = pwr.count_mults(A=sparse_A, maxiter=num_iter - 1)
    return xs, ys, f"sparsified, x = {x}, s = {s}; {num_mults:,} mults"

def expected_sparse_pwr(A, u_0, u_star, num_iter, seed, x):

    s_generator = SGenerator(A.shape[0], A.nnz)

    s = s_generator.get_min_s(x)

    xs, ys, lbl = sparse_pwr(A, u_0, u_star, num_iter, seed, s)
    return xs, ys, f"x = {x} {lbl}"

def percent_sparse_pwr(A, u_0, u_star, num_iter, seed, p):

    
    s_generator = SGenerator(A.shape[0], A.nnz)

    s = s_generator.proportion_sparse_s(p)

    xs, ys, lbl = sparse_pwr(A, u_0, u_star, num_iter, seed, s)

    return xs, ys, f"{p}%, {lbl}"