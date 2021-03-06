""" Computational Biology I: Quantitative Data Analysis
                Fruehjahrsemester 2019
                       Exercise 2
"""

import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt

def log_factorial(n):
    return(gammaln(n+1.0))

def log_binomial_coeff(n,k):
    """
    Call:
        lbc = log_binomial_coeff(n,k)
    Input argument:
        n: integer
        k: integer
    Output argument:
        lbc: float
    Example:
        log_binomial_coeff(10,5)
        =>
        5.529429087511423
    """
    lbc = log_factorial(n) - (log_factorial(k) + log_factorial(n-k))

    return(lbc)

def log_hypergeometricPMF(N,K,n,k):
    """
    Call:
        pmf = log_hypergeometricPMF(N,K,n,k)
    Input argument:
        N: integer
        K: integer
        n: integer
        k: integer
    Output argument:
        pmf: float
    Example:
        log_hypergeometricPMF(9,8,7,6)
        =>
        -0.2513144282809048
    """
    pmf = (log_binomial_coeff(K, k) + log_binomial_coeff(N-K, n-k)) - log_binomial_coeff(N, n)

    return(pmf)

def hypergeometricCDF(N,K,n,x):
    """
    Call:
        p = hypergeometricCDF(N,K,n,x)
    Input argument:
        N: integer
        K: integer
        n: integer
        x: integer
    Output argument:
        p: float
    Example:
        hypergeometricCDF(120,34,12,7)
        =>
        0.995786
    """

    p = np.sum(np.exp(log_hypergeometricPMF(N, K, n, np.arange(x+1))))

    return(p)

def hypergeometricPPF(N,K,n,q):
    """
    Call:
        k = hypergeometricPPF(N,K,n,q)
    Input argument:
        N: integer
        K: integer
        n: integer
        q: float
    Output argument:
        k: integer
    Example:
        hypergeometricPPF(120,34,12,0.25)
        =>
        2
    """

    cdf = 0
    k = -1
    while cdf < q:
        k += 1
        cdf += np.exp(log_hypergeometricPMF(N,K,n,k))

    return(k)



def hypergeometricIQR(N,K,n):
    """
    Call:
       p = hypergeometricIQR(N,K,n)
    Input argument:
       N: integer
       K: integer
       n: integer
    Output argument:
       iqr: float
    Example:
       hypergeometricIQR(1000,700,100)
       =>
       6.0
    """

    iqr = hypergeometricPPF(N,K,n,0.75) - hypergeometricPPF(N,K,n,0.25)

    return(iqr)

def approximateHypergeometric(N,K,n):
    """
    Call:
        p1,p2 = approximateHypergeometric(N,K,n)
    Input argument:
       N: integer
       K: integer
       n: integer
    Output argument:
       p1: array of length n (true PMF)
       p2: array of length n (approximate PMF)
    Example:
       approximateHypergeometric(1200,340,12)
       =>
       array([  1.79593325e-02,   8.63063328e-02,   1.89315480e-01,
         2.50640940e-01,   2.23061612e-01,   1.40583731e-01,
         6.43381828e-02,   2.15428101e-02,   5.23784737e-03,
         9.01836031e-04,   1.04373331e-04,   7.29033743e-06,
         2.32414827e-07]),
       array([  1.83572010e-02,   8.70899767e-02,   1.89370066e-01,
         2.49557451e-01,   2.21990058e-01,   1.40421618e-01,
         6.47681106e-02,   2.19479976e-02,   5.42319709e-03,
         9.52913183e-04,   1.13019936e-04,   8.12405457e-06,
         2.67652961e-07])
    """
    p = K/N
    k = np.arange(n+1)
    p1 = np.exp(log_hypergeometricPMF(N, K, n, k))
    p2 = np.exp(log_binomial_coeff(n,k)) * p**k * (1-p)**(n-k)

    # plt.plot(np.arange(0, n + 1), p2-p1)
    # plt.title("Error")
    # plt.show()

    return(p1,p2)

def cellCulture1(n,m,N=1000):
    """
    Call :
       p = cellCulture1(n,m,N=1000)
    Input argument:
       n: integer
       m: integer
       N: integer (optional, default=1000)
    Output argument:
       p: numpy array of length N
    Example:
       cellCulture1(10,3)
       =>
       p:
       array([  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                7.93587968e-09,   3.15206455e-08,   7.82477871e-08,
                1.55394600e-07,   2.70025476e-07,   4.28995157e-07,
	         ...
        	2.42875364e-16,   5.41360210e-17,   6.78750869e-18,
      		0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
    	        0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
      		0.00000000e+00])
    """

    a = np.exp(log_hypergeometricPMF(N, np.arange(N+1), n, m))
    p = a / np.sum(a[m:N-(n-m)])

    return(p)

def plot_cellCulture1(n,m,N=1000):
    """
    Call :
       plot_cellCulture1(n,m,N=1000)
    Input argument:
       n: integer
       m: integer
       N: integer (optional, default=1000)
    Output:
        Plot
    """
    plt.plot(np.arange(0,N+1),cellCulture1(n,m,N))
    plt.show()
    return

def cellCulture2(n,m,N=1000):
    """
    Call :
       M_min, M_max = cellCulture2(n,m,N=1000)
    Input argument:
       n: integer
       m: integer
       N: integer (optional, default=1000)
    Output argument:
       M_min: integer
       M_max: integer
    Example:
       cellCulture2(10,3)
       =>
       M_min: 110
       M_max: 608
    """
    cdf = np.cumsum(cellCulture1(n, m, N))
    M_min = np.argmax(cdf > 0.025) # returns first true
    M_max = np.argmin(cdf <= 0.975) # returns first false

    return(M_min,M_max)

def cellCulture3(n,m,N=1000):
    """
    Call :
       e = cellCulture3(n,m,N=1000)
    Input argument:
       n: integer
       m: integer
       N: integer (optional, default=1000)
    Output argument:
       e: float
    Example:
       cellCulture3(10,3)
       =>
       1.3871866295264623
    """

    M_min, M_max = cellCulture2(n, m, N)
    e = (M_max - M_min) / (0.5*(M_min + M_max))

    return(e)

def cellCulture4(eps,N=1000):
    """
    Call :
       n,e = cellCulture4(eps,N=1000)
    Input argument:
       eps: float, error threshold
       N: integer (optional, default=1000)
    Output argument:
       n: integer, minimal experiment number
       e: float, relative error
    Example:
       cellCulture4(0.1)
       =>
       n: 611
       e: 0.098098098098098094
    """

    n = 0
    e = eps + 1
    while e >= eps:
        n += 1
        e = cellCulture3(n,(n//2),N)

    return(n,e)