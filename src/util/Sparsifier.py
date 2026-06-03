"""
Based on Thm 1.5 in: https://doi.org/10.1145/1219092.1219097

Sparsifying matrix based on its entries, with higher probability of keeping
entries with higher magnitude
"""
#TODO: this is sparsifying drastically

import numpy as np

class Sparsifier():
    """
    A direct implementation of Thm 1.5
    """
    def __init__(self, seed):
        self.seed(seed)

    def seed(self, seed):
        """ Set the seed of the random number generator
        """
        self.rng = np.random.default_rng(seed=seed)

    def p_valid(self, p):
        """ Check if p is valid for a matrix with given dimensions
        Args:
            p: sparsification probability

        Return: True if valid, False if not
        """

        if p <= 0 or p > 1:
            # p out of bounds
            # print(f"INVALID p, {p} is not in (0, 1]")
            return False
        
        return True
    
    def prob_error(self, n):
        """ Get the probability of error associated with the sparsification
        process

        Thm 1.5 (1)
            probability of validity is: 1 - exp(-19(log n)^4)
            probability of error is: exp(-19(log n)^4)

        Args:
            n: the number of nonzeroes in the matrix being sparsified? TODO: confirm

        Return: the probability of error
        """
        return np.exp(-19 * (np.log(n))**4)
    
    def exp_nnz_bound(self, A, p, m, n, b):
        """ Get the expected number of nonzeroes in the sparsified matrix

        Thm 1.5 (2):
            (pmn) x Avg[(A_ij /b)^2] + m(8 log n)^4

        Args:
            A: the matrix being sparsified
            p: the sparsification factor
            m: the number of rows in the matrix
            n: the number of columns in the matrix
            b: the maximum absolute value of an entry in the matrix

        Return: the expected number of nonzeroes in the sparsified matrix
        """
        
        avg_sqr = np.mean((A.data / b) ** 2) 
        return p * m * n * avg_sqr + m * (8 * np.log(n)) ** 4

    def sparse_entry(self, x, p):
        """
        Scale value or make value zero

        Args: 
            x: the value to either scale up or make zero
            p: the probability of keeping the entry

        RETURN: x scaled up by a factor of s, zero otw
        """
        r = self.rng.random() # r in range [0.0, 1.0)

        if r < p:
            # With probability p scale the entry up
            return x / p
        else: 
            # With probability 1 - p make the entry zero
            return 0.0
        

    def sparsify(self, A, p):
        ''' Implementation of Thm 1.5 in: 
        https://doi.org/10.1145/1219092.1219097

        Increasing p => make more sparse 

        Sparsify a matrix A given some value p
        
        Args: 
            A: the matrix to sparsify
            p: factor of sparsification

        Return: NONE, sparsifies A in place
        '''

        # b is the maximum absolute value of an entry in A
        b = abs(A).max()

        # Number of nonzeroes in A
        nnz = A.nnz

        # Approx dimensions of a dense matrix with nnz nonzeroes
        sqrt_nnz = np.sqrt(nnz)
        #TODO is this correct? or should it just be nxm?
        n_approx = np.ceil(sqrt_nnz) 
        m_approx = np.floor(sqrt_nnz)

        if (m_approx < 76):
            print(f"WARNING: m_approx {m_approx} is less than 76, sparsification may not be valid")

        if self.prob_error(n=n_approx) > 0.01:
            print(f"WARNING: probability of error {self.prob_error(n=n_approx)} is greater than 0.01, sparsification may not be valid")

        for i, datum in enumerate(A.data):
            importance = p * (datum / b) ** 2
            prob_sparse = max(importance,  np.sqrt(importance * (8 * np.log(n_approx)) ** 4 / 4))
            A.data[i] = self.sparse_entry(x=datum, p=prob_sparse)

        A.eliminate_zeros()
        
        exp_nnz_bound = self.exp_nnz_bound(A=A, p=p, m=m_approx, n=n_approx, b=b)
        # print(f"Expected number of nonzeroes in sparsified matrix is at most {exp_nnz_bound}")
        act_nnz = nnz - A.nnz

        if (act_nnz > exp_nnz_bound):
            raise ValueError(f"Actual number of nonzeroes in sparsified matrix {act_nnz} is greater than expected bound {exp_nnz_bound}, sparsification may not be valid")

        return None
    

