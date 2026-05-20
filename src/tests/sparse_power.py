from Sparsification_Research.src.MDSparsifier import MDSparsifier
from Sparsification_Research.src.SGenerator import SGenerator

import numpy as np

from ..util.eig_functs import euclidean_dist
from ..util import power as pwr

def sparse_pow(A, u_0, u_star, num_iter, seed, x):
    sparsifier = MDSparsifier(seed=seed)

    s_generator = SGenerator(A.shape[0], A.nnz)

    xs = np.zeros(num_iter)
    ys = np.zeros(num_iter)

    s = s_generator.get_min_s(x)

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
    num_mults = pwr.count_mults(A=sparse_A, maxiter=num_iter)
    return xs, ys, f"sparsified, x = {x}, s = {s}; {num_mults} mults"