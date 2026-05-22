from .util.JohnsonLindenstrauss import JohnsonLindenstrauss
import numpy as np
from scipy import linalg
import sys
"""
TODO: 
(1) Code up some Johnson Lindenstrauss algorithm(s)?
(2) Is the top left eigenvector of Johnson Lindenstrauss similar to original? 
"""

from Sparsification_Research.src.SSGetter import SSGetter
from Sparsification_Research.src.old_code.MatrixChecker import MatrixChecker


def test_one(eps, d):
    print("Test 1\n")
    MATS = ["662_bus"]
    num_xs = 1

    
    ssgetter = SSGetter()

    mats = ssgetter.get_by_name(names=MATS)

    jonny = JohnsonLindenstrauss()
    
    mc = MatrixChecker()
    for name, A in mats.items():
        xs = np.random.rand(num_xs, A.shape[0]) # randomly generated x vectors
        for x in xs:
            print(f"{name}: {A.shape[0]}x{A.shape[0]}")
            approx,orig = jonny.get_b(A,x,eps,d)

            approx_norm = np.linalg.norm(approx, ord=2)
            orig_norm = np.linalg.norm(orig,ord=2)
            print(abs(approx_norm - orig_norm)/orig_norm)

    return jonny

def eig_top_right(A):
  eigenvalues,a_ev = linalg.eig(A)
  # Get index of largest eigenvalue (by magnitude)
  top_idx = np.argmax(np.abs(eigenvalues))
  # Top right eigenvector
  a_ev_tr = a_ev[:, top_idx]
  #check for negative eigenvalue
  if(eigenvalues[top_idx]<0):
        print(f"found negative eigenvalue {eigenvalues}, negating associated vector")
        a_ev_tr *=-1
  return a_ev_tr

def test_two(eps, d):
    MATS = ["662_bus"]
    num_xs = 1

    
    ssgetter = SSGetter()

    mats = ssgetter.get_by_name(names=MATS)
    jonny = test_one(eps,d)
    print("Test 2\n")
    mc = MatrixChecker()
    for name, A in mats.items():
        xs = np.random.rand(num_xs, A.shape[0]) # randomly generated x vectors
        for x in xs:
            print(f"{name}: {A.shape[0]}x{A.shape[0]}")
            P = jonny.jl_matrix()
            
            nearI = P.transpose() @ P
            print(f"norm of P^T *P - I {np.linalg.norm(nearI - np.eye(nearI.shape[0]))}")

            nearI = P @ P.transpose()
            print(f"norm of P*P^T - I {np.linalg.norm(nearI - np.eye(nearI.shape[0]))}")
            
            approx = P @ A @ P.transpose()
            orig = A
            print(f"approx: {approx.shape[0]}x{approx.shape[0]}")
            print(f"orig: {orig.shape[0]}x{orig.shape[0]}")
            approx_eigenvalues = eig_top_right(approx)
            orig_eigenvalues = eig_top_right(orig.todense())

            #project eigenvalues
            lowdimdiff = (P @ orig_eigenvalues) - approx_eigenvalues
            highdimdiff = orig_eigenvalues - (P.transpose()@approx_eigenvalues)
            lowdimdiffnorm = np.linalg.norm(lowdimdiff, ord=2)
            highdimdiffnorm = np.linalg.norm(highdimdiff, ord=2)
            print(f'lowdim norm of eig error {lowdimdiffnorm}')
            print(f'highdim norm of eig error {highdimdiffnorm}')

    return jonny

def sparsen(A,density):
  import random
  mat = A.copy()
  nz = 0
  for i in range(A.shape[0]):
    for j in range(A.shape[1]):
      r = random.uniform(0,1)
      if r > density:
        mat[i,j] =0
        nz += 1
      elif mat[i,j] == 0:
        nz +=1
  nnz = mat.shape[0]*mat.shape[1] - nz
  return mat,nnz

def test_three(eps, d):
    MATS = ["662_bus"]
    num_xs = 1

    
    ssgetter = SSGetter()

    mats = ssgetter.get_by_name(names=MATS)
    jonny = test_two(eps,d)
    print("Test 3\n")
    mc = MatrixChecker()
    for name, A in mats.items():
        xs = np.random.rand(num_xs, A.shape[0]) # randomly generated x vectors
        print(f" A nnz {A.nnz}, A entries {A.shape[0]*A.shape[1]}")
        density = A.nnz/(A.shape[0] * A.shape[1])

        for x in xs:
            print(f"{name}: {A.shape[0]}x{A.shape[0]}")
            P = jonny.jl_matrix() 
            approx = P @ A @P.transpose()
            sparse,sparsennz = sparsen(approx, density)
            
            print(f" sparsified nnz {sparsennz}, A entries {sparse.shape[0]*sparse.shape[1]}")
            approx_norm = np.linalg.norm(approx-sparse, ord=2)
            print(f"norm of PA - sparsen(PA) {approx_norm}")

            a_ev_tr = eig_top_right(approx)
            
            s_ev_tr = eig_top_right(sparse)
            
            eigendiff = a_ev_tr - s_ev_tr
            eigendiff_norm = np.linalg.norm(eigendiff, ord=2)
            print(f"norm of eig(PAP^T) - eig(sparsen(PAP^T)) {eigendiff_norm}")
            orig = A
            orig_eigenvalues = eig_top_right(orig.todense())
            norm_oe = np.linalg.norm(orig_eigenvalues,ord=2)

            #project eigenvalues
            lowdimdiff = (P @ orig_eigenvalues) - s_ev_tr
            highdimdiff = orig_eigenvalues - (P.transpose()@ s_ev_tr)
            lowdimdiffnorm = np.linalg.norm(lowdimdiff, ord=2)
            highdimdiffnorm = np.linalg.norm(highdimdiff, ord=2)
            print(f'lowdim norm of P*eig(A) - eig(sparse) {lowdimdiffnorm}')
            print(f'highdim norm of eig(A) - P^T*eig(sparse) {highdimdiffnorm}')
            print(f'highdim norm {highdimdiffnorm} / norm(eig(A)) {norm_oe}={highdimdiffnorm/norm_oe}')
    return jonny

def test_four(eps, d):
    '''Test sparsening JL projection matrix P'''
    MATS = ["662_bus"]
    num_xs = 1

    
    ssgetter = SSGetter()

    mats = ssgetter.get_by_name(names=MATS)
    jonny = test_three(eps,d)
    print("Test 4\n")
    mc = MatrixChecker()
    for name, A in mats.items():
        xs = np.random.rand(num_xs, A.shape[0]) # randomly generated x vectors
        print(f" A nnz {A.nnz}, A entries {A.shape[0]*A.shape[1]}")
        density = A.nnz/(A.shape[0] * A.shape[1])

        for x in xs:
            print(f"{name}: {A.shape[0]}x{A.shape[0]}")
            P,Pnnz = sparsen(jonny.jl_matrix(), density)
        
            print(f" P nnz {Pnnz}, P entries {P.shape[0]*P.shape[1]}")
            approx = P @ A @P.transpose()
            sparse,sparsennz = sparsen(approx, density)
            
            nearI = P.transpose() @ P
            print(f"norm of sparsen(P)^T *sparsen(P) - I {np.linalg.norm(nearI - np.eye(nearI.shape[0]))}")

            nearI = P @ P.transpose()
            print(f"norm of sparsen(P)*sparsen(P)^T - I {np.linalg.norm(nearI - np.eye(nearI.shape[0]))}")
           
            Pinv = np.linalg.pinv(P)
            print(f"norm of sparsen(P)^+ *sparsen(P) - I {np.linalg.norm(Pinv@P - np.eye(Pinv.shape[0]))}")

            print("\nP appears to be a near orthogonal matrix, and the peuedo-inverse\
                \nis only 50% better than P^T as measured by the distance from the identity\
                \nreferring back to test 2 we also see that sparsening the projection matrix P\
                \nmakes P more orthogonal. recall an orthogonal matrix has the property\
                \nAA^T=I.")
            print(f" sparsen(P)A nnz {sparsennz}, sparsen(P)A entries {sparse.shape[0]*sparse.shape[1]}")

            s_ev_tr = eig_top_right(sparse)
            
            orig = A
            orig_eigenvalues = eig_top_right(orig.todense())
            norm_oe = np.linalg.norm(orig_eigenvalues,ord=2)

            #project eigenvalues
            lowdimdiff = (P @ orig_eigenvalues) - s_ev_tr
            highdimdiff = orig_eigenvalues - (P.transpose()@ s_ev_tr)
            lowdimdiffnorm = np.linalg.norm(lowdimdiff, ord=2)
            highdimdiffnorm = np.linalg.norm(highdimdiff, ord=2)
            print(f'lowdim norm of sparsen(P)*eig(A) - eig(sparsen(P)A) {lowdimdiffnorm}')
            print(f'highdim norm of eig(A) - sparsen(P)^T*eig(sparsen(P)A) {highdimdiffnorm}')
            print(f'highdim norm {highdimdiffnorm} / norm(eig(A)) {norm_oe}={highdimdiffnorm/norm_oe}')
    return jonny


def jl_validity(eps, d):
    # MATS = ["494_bus", "662_bus", "685_bus", "1138_bus", "bcsstk21", "bcsstm25", "bcsstm39", "finan512", "jnlbrng1", "m3plates"] #some matrices that converged w/ Jacobi
    MATS = ["662_bus"]
    NUM_XS = 10

    
    ssgetter = SSGetter(in_csr=False, row_bounds=(600, 700))

    # mats = ssgetter.get_next(1)
    mats = ssgetter.get_by_name(MATS)
    jonny = JohnsonLindenstrauss()
    
    for name, A in mats.items():
        print(f"{name}: {A.shape[0]}x{A.shape[0]}")
        xs = np.random.rand(NUM_XS, A.shape[0]) # randomly generated x vectors
        A_reduced = jonny.reduce(A, eps, d)

        print(f"A_reduced.shape = {A_reduced.shape}")

        res = []
        for x in xs:
            b_approx = A_reduced @ x
            b = A @ x

            approx_norm = np.linalg.norm(b_approx, ord=2)
            orig_norm = np.linalg.norm(b,ord=2)
            res.append(abs(approx_norm - orig_norm)/orig_norm)   
        print(f"average distance of {NUM_XS} randomly generated vectors {np.mean(res)}")   

if __name__ == "__main__":
    epsilon = 1/64
    
    n = 8
    ds = [492,493,494]
    mats = ["494_bus"] 
    if(len(sys.argv)==0):
        test = Tester(jl,mats=mats, save_fig=False, show_fig=True)
        test.compare_eigenvectors(ds, epsilon, n)
    
    d = 500
    d = 50
    test_four(epsilon, d)
    jl_validity(epsilon, d)
