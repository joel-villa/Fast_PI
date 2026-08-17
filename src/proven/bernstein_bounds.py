"""The following file contains Bernstein Concentration Bound results for the 
quantity ||~Av||, where ~A is the samples version of A, and v is the normalized 
top eigenvector of A

Assumptions: 
(1) The maximum row magnitude is 1
(2) The row magnitudes WLOG are upperbounded by a powerlaw distribution i.e.
    ||a_i|| = c * i^-k, for some constants c, k > 0
(3) The variance proxy is positive
(4) A is PSD, i.e. ||Av|| = lambda 
"""
