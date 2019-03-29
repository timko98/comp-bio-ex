"""
Computational Biology I: Quantitative Data Analysis
                Fruehjahrsemester 2019
                       Exercise 4
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def uniformPDF(x,a=0.0,b=4.0):
    p = 1.0/(b-a)*np.ones((len(x)))
    p[x<a] = 0.0
    p[x>b] = 0.0
    return(p)

def exponentialPDF(x,a=1.0):
    p = a*np.exp(-a*x)
    p[x<0] = 0.0
    return(p)

def paretoPDF(x,b=2.0):
    p = b/(x**(b+1.0))
    p[x<1] = 0.0
    return(p)

def uniformRVS(m,a,b):
    rvs = a + stats.uniform.rvs(size=m)*(b-a)
    return(rvs)

def exponentialRVS(m,a):
    rvs = stats.expon.rvs(scale=1.0/a,size=m)
    return(rvs)

def paretoRVS(m,b):
    rvs = stats.pareto.rvs(b,size=m)
    return(rvs)

def normalRVS(m,mu=0.0,sigma2=1.0):
    rvs = stats.norm.rvs(loc=mu,scale=np.sqrt(sigma2),size=m)
    return(rvs)

def uniformMean(m,a=0.0,b=4.0):
    """
    Call:
       s = uniformMean(m,a,b)
    Input argument:
       m: integer
       a: float, default = 0.0
       b: float, default = 4.0
    Output argument:
       s: float
    Examples:
       In [1]: uniformMean(2)
       Out[1]: 3.3903568167258782

       In [2]: uniformMean(2)
       Out[2]: 1.7787091976146681
    """
    s = np.mean(uniformRVS(m,a,b))
    return(s)

def exponentialMean(m,a=1.0):
    """
    Call:
       s = exponentialMean(m,a)
    Input argument:
       m: integer
       a: float, default = 1.0
    Output argument:
       s: float
    Examples:
       In [1]: exponentialMean(2)
       Out[1]: 1.2127541524661218

       In [2]: exponentialMean(2)
       Out[2]: 1.6322926962391706
    """
    s = np.mean(exponentialRVS(m,a))
    return(s)

def paretoMean(m,b=2.0):
    """
    Call:
       s = paretoMean(m,b)
    Input argument:
       m: integer
       b: float, default = 2.0
    Output argument:
       s: float
    Examples:
       In [1]: paretoMean(2)
       Out[1]: 1.3636383632826559

       In [2]: paretoMean(2)
       Out[2]: 1.3419141482243173
    """
    s = np.mean(paretoRVS(m,b))
    return(s)

def uniformMeans(n,m,a=0.0,b=4.0):
    """
    Call:
       v = uniformMeans(n,m,a,b)
    Input argument:
       n: integer
       m: integer
       a: float, default = 0.0
       b: float, default = 4.0
    Output argument:
       v: numpy array
    Examples:
       In [1]: uniformMeans(4,2)
       Out[1]: array([ 1.83283611,  0.84751247,  2.79956065,  2.08173557])

       In [2]: uniformMeans(4,2)
       Out[2]: array([ 2.20206943,  2.84842631,  1.44178502,  2.43703912])
    """
    v = np.zeros(n)
    for i in range(n):
        v[i] = uniformMean(m,a,b)
    return(v)

def exponentialMeans(n,m,a=1.0):
    """
    Call:
       v = exponentialMeans(n,m,a)
    Input argument:
       n: integer
       m: integer
       a: float, default = 1.0
    Output argument:
       v: numpy array
    Examples:
       In [1]: exponentialMeans(4,2)
       Out[1]: array([ 1.97798574,  2.83550965,  1.17162626,  1.83645093])

       In [2]: exponentialMeans(4,2)
       Out[2]: array([ 2.11823173,  1.33106409,  1.55833024,  1.36089404])
    """
    v = np.zeros(n)
    for i in range(n):
        v[i] = exponentialMean(m,a)
    return(v)

def paretoMeans(n,m,b=2.0):
    """
    Call:
       v = parteoMeans(n,m,b)
    Input argument:
       n: integer
       m: integer
       b: float, default = 2.0
    Output argument:
       v: numpy array
    Examples:
      In [1]: paretoMeans(4,2)
      Out[1]: array([ 1.52811783,  1.14560974,  1.10754114,  1.8280829 ])

      In [2]: paretoMeans(4,2)
      Out[2]: array([ 2.27185097,  1.57569186,  3.32299622,  1.7457624 ])
    """
    v = np.zeros(n)
    for i in range(n):
        v[i] = paretoMean(m,b)
    return(v)

def plot_DistributionMeans(dist):
    """
    Call:
       plot_DistributionMeans(dist)
    Input argument:
       dist: distrubution to plot
           0 : uniform (a=0, b=4)
           1 : exponential (a=1)
           2 : pareto (b=2)
    """
    # generate a 1000 samples of means of 2,3 and 10 uniform RVS each
    M = 1000
    if dist == 0:
        u2 = uniformMeans(M,2)
        u3 = uniformMeans(M,3)
        u10 = uniformMeans(M,10)
    elif dist == 1:
        u2 = exponentialMeans(M,2)
        u3 = exponentialMeans(M,3)
        u10 = exponentialMeans(M,10)
    elif dist == 2:
        u2 = paretoMeans(M,2)
        u3 = paretoMeans(M,3)
        u10 = paretoMeans(M,10)
    # plot as histograms
    plt.hist(u2,25,histtype='step',edgecolor='blue',facecolor='none',linewidth=2)
    plt.hist(u3,25,histtype='step',edgecolor='green',facecolor='none',linewidth=2)
    plt.hist(u10,25,histtype='step',edgecolor='red',facecolor='none',linewidth=2)
    plt.show()
    plt.clf()

def fourCumulants(x):
    """
    Call:
       c1,c2,c3,c4 = fourCumulants(x)
    Input argument:
       x: numpy array
    Output argument:
       c1: float
       c2: float
       c3: float
       c4: float
    Examples:
       In [1]: u = uniformMeans(50,2)

       In [2]: fourCumulants(u)
       Out[195]: 
       (2.2088148839791724,
       0.67105003125814866,
       -0.072068671445580129,
       -0.5057953093797507)

       In [3]: u = uniformMeans(50,2)

       In [4]: fourCumulants(u)
       Out[4]: 
       (2.0452126992181343,
       0.53514435619719569,
       -0.051635297540346875,
       -0.095295067366394037)
    """
    c1 = np.mean(x)
    c2 = np.mean((x-c1)**2)
    c3 = np.mean((x-c1)**3)
    c4 = np.mean((x-c1)**4) - 3*(np.mean(x*x)-c1**2)**2
    return(c1,c2,c3,c4)

def cumulantsOfMeans(dist,M,n=10000):
    """
    Call:
       v = cumulantsOfMeans(dist,M,n)
    Input argument:
       dist: integer specifying the distribution. 0=uniform, 1=exponential, 2=pareto
       M: integer
       n: integer
    Output argument:
       C: a 4-by-M numpy array. Nth row = Nth cumulant for the mean over 1...M random values. 
    Examples:
       In [1]: cumulantsOfMeans(0,3)
       Out[1]: 
       array([[ 2.00058641,  1.99235859,  2.00111604],
              [ 1.33388974,  0.67359295,  0.43723978],
              [-0.00967865,  0.01280984,  0.00513898],
              [-2.12997463, -0.28156999, -0.07535785]])

       In [2]: cumulantsOfMeans(0,5)
       Out[2]: 
       array([[ 2.00008167,  2.01331328,  2.00421357,  2.00617495,  2.00430073],
              [ 1.32904624,  0.67287942,  0.44120348,  0.33495163,  0.27132572],
              [-0.00855118, -0.01235707, -0.00283194, -0.00584591, -0.00739828],
              [-2.1174904 , -0.28100373, -0.07818415, -0.03272124, -0.01731555]])
    """
    C = np.zeros((4,M))
    if dist == 0:
        for i in range(M):
            C[:,i] = fourCumulants(uniformMeans(n,i+1))
    if dist == 1:
        for i in range(M):
            C[:,i] = fourCumulants(exponentialMeans(n,i+1))
    if dist == 2:
        for i in range(M):
            C[:,i] = fourCumulants(paretoMeans(n,i+1))
    return(C)

def plot_cumulantsOfMeans(dist):
    M = 100
    # Generate your matrix of cumulants :
    C = cumulantsOfMeans(dist,M)
    # plot each of your 4 cumulants as a function of m=1,...,M
    if dist == 0 or dist == 1:
        c1, = plt.plot(np.arange(M)+1,C[0,:])
        c2, = plt.plot(np.arange(M)+1,C[1,:])
        c3, = plt.plot(np.arange(M)+1,C[2,:])
        c4, = plt.plot(np.arange(M)+1,C[3,:])
    else:
        c1, = plt.semilogy(np.arange(M)+1,C[0,:])
        c2, = plt.semilogy(np.arange(M)+1,C[1,:])
        c3, = plt.semilogy(np.arange(M)+1,C[2,:])
        c4, = plt.semilogy(np.arange(M)+1,C[3,:])
    plt.legend([c1,c2,c3,c4],["mean","variance","skewness","kurtosis"])
    plt.show()
    plt.clf()

def zScore(x):
    """
    Call:
       z = zScore(x)
    Input argument:
       v: numpy array
    Output argument:
       z: numpy array
    Examples:
       In [1]: u = uniformMeans(5,2)

       In [2]: zScore(u)
       Out[2]: array([-1.24665979, -0.29325619,  1.21160404,  1.1227908 , -0.79447886])

       In [1]: u = uniformMeans(5,2)
       
       In [2]: zScore(u)
       Out[2]: array([ 0.04043301, -0.88987265, -1.03062589,  0.11028011,  1.76978542])
    """
    z = (x-np.mean(x))/np.sqrt(np.var(x))
    return(z)

def gaussianApproximation(n,me=0.0,sd=1.0):
    """
    Call:
       v = gaussianApproximation(n,me,sd)
    Input argument:
       n: integer
      me: float
      sd: float
    Output argument:
       v: numpy array
    Examples:
       In [1]: gaussianApproximation(10000,3,0.1)
       Out[1]: 
       array([ 3.25481318,  3.20421592,  2.97242592, ...,  2.91381843,
               3.07523275,  2.9385334 ])

       In [2]: gaussianApproximation(10000,3,0.1)
       Out[2]: 
       array([ 2.86644104,  3.03176111,  3.12001852, ...,  2.89964535,
               2.97468303,  2.93763211])
    """
    x = zScore(uniformMeans(n,10)) * sd + me
    return(x)

def plot_gaussianApproximation(n,me=0.0,sd=1.0):
    x1 = gaussianApproximation(n,me,sd)
    x2 = normalRVS(n,me,sd)
    plt.hist(x1,50,histtype='step',edgecolor='blue',facecolor='none',linewidth=2)
    plt.hist(x2,50,histtype='step',edgecolor='green',facecolor='none',linewidth=2)
    plt.show()
    plt.clf()

