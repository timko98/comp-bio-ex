""" Computational Biology I: Quantitative Data Analysis
                Fruehjahrsemester 2019
                       Exercise 3
"""

import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt


def log_factorial(n):
    return(gammaln(n+1.0))

    
def log_binomial_coeff(n,k):
    lbc = log_factorial(n) - log_factorial(k) - log_factorial(n-k)
    return(lbc)


def hypergeometricPMF(N,K,n,k):
    pmf = np.exp(log_binomial_coeff(N-K,n-k) + log_binomial_coeff(K,k) - log_binomial_coeff(N,n))
    return(pmf)


def hypergeometricCDF(N,K,n,x): 
    cdf = sum(hypergeometricPMF(N,K,n,np.arange(x+1)))
    return(cdf)


def cellCulture5(n,m,N=1000):
    """
    Call:
       M_min, M_max = cellCulture5(n,m,N=1000)
    Input argument:
       n: integer
       m: integer
       N: integer (optional, default=1000)
    Output arguments:
       M_min: integer
       M_max: integer
    Example:
       In [1]: cellCulture5(7,2,500)
       Out[1]: (19, 353)

       In [2]: cellCulture5(10,3,1000)
       Out[2]: (68, 651)
    """
    min, max = np.zeros(N+1), np.zeros(N+1)

    for M in range(N+1):
        max[M] = np.sum(hypergeometricPMF(N, M, n, np.arange(m+1)))
        min[M] = np.sum(hypergeometricPMF(N, M, n, np.arange(m, N+1)))

    M_min = np.argmax(min > 0.025)
    M_max = np.argmin(max > 0.025)-1
    
    return(M_min,M_max)
   

def likelihood(N,n,k):
    """
    Call:
        l = likelihood(N,n,k)
    Input argument:
        N: integer (array)
        n: integer (array)
        k: integer (array)
    Output argument:
        l: float (array)
    Example:
        likelihood(6,10,5)
        =>
        1.190748e-05
    """
    # np.seterr(all='ignore')
    #
    # N = np.array(N, dtype=np.float64)
    # n = np.array(n, dtype=np.float64)
    # k = np.array(k, dtype=np.float64)

    l = np.exp((log_factorial(N) - log_factorial(N-k)) - n*np.log(N))

    return(l)


def posterior(N,n,k,Nmax=10000):
    """
    Call:
        p = posterior(N,n,k,Nmax)
    Input argument:
        N: integer (array)
        n: integer
        k: integer
        Nmax: integer (default=10000)
    Output argument:
        p = float (array)
    Examples:
        posterior(70,100,50)
        => 
        0.030743
        
        posterior(np.arange(1,100),100,50)
        =>
        array([  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 ...
                 4.02034494e-05,   2.94577274e-05,   2.15749533e-05,
                 1.57969637e-05,   1.15644322e-05,   8.46547904e-06])
    """
    post = np.array(likelihood(N,n,k) / np.sum(likelihood(np.arange(k,Nmax+1), n, k)))
    post[np.array(N) > Nmax] = 0

    return(post)


def plot_posterior(N,n,k):
    """
    Call :
       plot_posterior(N,n,k)
    Input argument:
       n: integer
       m: integer
       N: array
    Output:
        Plot
    Examples:
        plot_posterior(np.arange(1,100),100,50)
    """
    plt.plot(N,posterior(N,n,k))
    plt.show()
    return


def cdf(n,k,Nmax=10000):
    """
    Call:
        c = cdf(n,k,Nmax)
    Input argument:
        n: integer
        k: integer
        Nmax: integer (default=10000)
    Output argument:
        c: float array
    Example:
        cdf(100,50)[0:100] # i.e. the first 100 values
        =>
        array([  0.00000000e+00, 0.00000000e+00,   0.00000000e+00,
                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                 ...
                 9.99956756e-01,   9.99968321e-01,   9.99976786e-01,
                 9.99982983e-01])
    """
    c = np.cumsum(posterior(np.arange(1, Nmax+1),n,k))

    return(c)

# Code to plot cdf for the first 100 values
# c = cdf(100,50)
# plt.plot(np.arange(0,100),c[0:100])
# plt.show()


def mode_median_mean(n,k,Nmax=10000):
    """
    Call:
        m = mode(n,k)
    Input argument:
        n: integer
        k = integer
        Nmax: integer (default=10000)
    Output argument:
        mode = integer
        median = integer
        mean = float
    Example:
        mode_median_mean(100,70)
        =>
        (130, 134, 137.3873952703959)
    """
    N = np.arange(1,Nmax+1)

    mode = np.argmax(posterior(N, n, k))+1
    median = np.argmax(cdf(n,k) >= 0.5)
    mean = np.sum(N*posterior(N,n,k))

    return(mode, median, mean)

def ppi(n,k,Nmax=10000):
    """
    Call:
        p = ppi(n,k,Nmax)
    Input argument:
        n: integer
        k: integer
        Nmax: integer (default=10000)
    Output arguments:
        Imin: integer
        Imax: integer
    Example:
        ppi(100,50)
        =>
        (54, 80)
    """

    Imin = np.argmin(cdf(n,k) <= 0.01) + 1
    Imax = np.argmax(cdf(n,k) >= 0.99) + 1

    return(Imin,Imax)

if __name__ == "__main__":
    print(posterior(70,100,50))