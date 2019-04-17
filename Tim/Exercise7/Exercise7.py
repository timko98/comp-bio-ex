""" Computational Biology I: Quantitative Data Analysis
                Fruehjahrsemester 2019
                       Exercise 7
"""

from pylab import * #This imports everything from pylab. 
import numpy as np


def exponentialPDF(x,a=1.0):
    p = a*exp(-a*x)
    p[x<0] = 0.0
    return(p)


def loadDeathData(fname):
    """
    Call:
       x = loadData(fname)
    Input argument:
       fname: string (file name)
    Output argument:
       x: numpy array
    Example:
       x = loadDeathData('data1.dat')
       =>
       x: array([  6.63538701e+00,   9.44460875e+01,   2.59735954e+01,
                   3.97496500e+00,   1.00406378e+01,   3.55371193e+00,
                   ...
                   1.84695562e+00,   1.50890586e+01,   3.89125961e+00,
                   5.44639002e+00])
    """
    data = []
    for line in open(fname):
        data.append(float(line))
    a = array(data,dtype=float)
    return(a)


def logLikelihood(x,rho,lam1,lam2):
    """
    Call:
       logLik = logLikelihood(x,rho,lam1,lam2)
    Input argument:
       x: numpy array
       rho: float
       lam1: float
       lam2: float
    Output argument:
       logLik: float
    Example:
       x = loadData('data1.dat')
       logLik = logLikelihood(x,0.8,0.5,0.05)
       =>
       logLik: -4070.616429
    """
    logLik = np.sum(np.log(rho*exponentialPDF(x, lam1)+(1-rho)*exponentialPDF(x, lam2)))
    return(logLik)


def EM(x,rho,lam1,lam2,eps=0.1):
    """
    Call:
       logLik,rho,lam1,lam2 = EM(x,rho,lam1,lam2)
    Input argument:
       x: numpy array
       rho: float
       lam1: float
       lam2: float
       eps: float, stopping criterion logLik[j]-logLik[j-1] < eps (optional, default=0.1)
    Output argument:
       logLik: numpy array
       rho: numpy array
       lam1: numpy array
       lam2: numpy array
    Example:
       x = loadData('data1.dat')
       logLik,rho,lam1,lam2 = EM(x,0.3,0.4,0.2)
       =>       
       logLik: array([-4945.27821142, -3709.50189945, -3698.8008059 , -3688.57180056,
                      ...
                      -3515.01828998, -3514.48789956, -3514.18538475, -3514.01899986,
                      -3513.93013209])
       rho: array([0.3, 0.192624, ..., 0.9424968, 0.9442385])
       lam1: array([0.4, 0.266520, ..., 0.1065613, 0.1061257])
       lam2: array([0.2, 0.052651, ..., 0.0079728, 0.0077882])
       (EM stops after 28 iterrations)
    """
    n = len(x)
    converged = False
    logLik = list()
    rho_list = list()
    rho_list.append(rho)
    lam1_list = list()
    lam1_list.append(lam1)
    lam2_list = list()
    lam2_list.append(lam2)

    while not converged:
        old_log_lik = logLikelihood(x, rho, lam1, lam2)

        L1 = exponentialPDF(x, lam1)
        L2 = exponentialPDF(x, lam2)

        p1_i = (rho * L1) / (rho * L1 + ((1 - rho) * L2))
        p2_i = ((1 - rho) * L2) / (rho * L1 + ((1 - rho) * L2))

        rho = np.sum(p1_i) / n
        lam1 = np.sum(p1_i) / np.sum(x * p1_i)
        lam2 = np.sum(p2_i) / np.sum(x * p2_i)

        rho_list.append(rho)
        lam1_list.append(lam1)
        lam2_list.append(lam2)

        new_log_lik = logLikelihood(x, rho, lam1, lam2)

        logLik.append(old_log_lik)
        if (np.abs(np.abs(new_log_lik) - np.abs(old_log_lik))) < eps:
            logLik.append(new_log_lik)
            converged = True

    return(np.array(logLik),np.array(rho_list),np.array(lam1_list),np.array(lam2_list))


def persistors(x,rho,lam1,lam2):
    """
    Call:
       idx = persistors(x,rho,lam1,lam2)
    Input argument:
       x: numpy array
       rho: float
       lam1: float
       lam2: float
    Output argument:
       idx: numpy array
    Examples:
       x = loadData('data1.dat')
       logLik,rho,lam1,lam2 = EM(x,0.3,0.4,0.2)
       idx = persistors(x,rho[-1],lam1[-1],lam2[-1])
       =>
       idx : array([  1,  14,  21,  63,  64,  83, 115, 118, 119, 155, 161, 172, 177,
                    ...
                    759, 766, 898, 915, 984, 985, 988, 990, 992])

       idx = persistors(x,1-rho[-1],lam2[-1],lam1[-1])
       =>
       idx : array([  1,  14,  21,  63,  64,  83, 115, 118, 119, 155, 161, 172, 177,
                    ...
                    759, 766, 898, 915, 984, 985, 988, 990, 992])
    """
    L1 = exponentialPDF(x, lam1)
    L2 = exponentialPDF(x, lam2)

    p1_i = (rho * L1) / (rho * L1 + ((1 - rho) * L2))
    p2_i = ((1 - rho) * L2) / (rho * L1 + ((1 - rho) * L2))

    if rho < 0.5:
        p = p2_i / (p1_i + p2_i)
    else:
        p = p1_i / (p1_i + p2_i)

    idx = np.argwhere(p < 0.5)
    return(idx)


def EMcovariance(x,rho,lam1,lam2):
    """
    Call:
       calcCovariance(x,rho,lam1,lam2)
    Input argument:
       x: numpy array
       rho: float
       lam1: float
       lam2: float
    Output arguments:
       C: numpy array (3-by-3 covariance matrix)
    Example:
       x = loadData('data1.dat')
       logLik,rho,lam1,lam2 = EM(x,0.3,0.4,0.2)
       C = EMcovariance(x,rho[-1],lam1[-1],lam2[-1])
       =>
       C:
       array([[  1.31617850e-04,  -1.91363013e-05,  -8.30007943e-06],
              [ -1.91363013e-05,   1.78489229e-05,   1.92546421e-06],
              [ -8.30007943e-06,   1.92546421e-06,   2.00836782e-06]])
    """

    return(C)


def parameterCorrelations(C):
    """
    Call:
       R,F = parameterCorrelations(C)
    Input argument:
       C: numpy array (3-by-3 covariance matrix)
    Output arguments:
       R: numpy array (3-by-3 correlation matrix)
       F: numpy array (identifiability vector)
    Example:
       x = loadData('data1.dat')
       logLik,rho,lam1,lam2 = EM(x,0.3,0.4,0.2)
       C = EMcovariance(x,rho[-1],lam1[-1],lam2[-1])
       R,F = parameterCorrelations(C)
       =>
       R:
       array([[ 1.        , -0.39481587, -0.51050879],
              [-0.39481587,  1.        ,  0.32159391],
              [-0.51050879,  0.32159391,  1.        ]])
       F:
       array([ 0.72652167,  0.76173   ,  0.73845743])
    """

    return(R,F)

#####
# helper functions for vizualizing the data/model
#####

def plotDeathCurve(x,ylog=False):
    """
    Call:
       plotKillCurve(x)
    Input argument:
       x: numpy array
    """
    bins = arange(0, ceil(max(x)))
    h,edge = histogram(x, bins, normed=True)
    kc = 1 - cumsum(h)
    plot(edge[1:], kc)
    if ylog:
        yscale('log')
    ylim(1.0 / len(x), 1)
    xlabel('Survival Time')
    ylabel('Frequency')
#plotDeathCurve(x)
#plotDeathCurve(x, ylog=True)


def plotDataAndModel(x,rho,lam1,lam2):
    """
    Call:
       plotDataAndModel(x,rho,lam1,lam2)
    Input argument:
       x: numpy array
       rho: float
       lam1: float
       lam2: float
    """
    xx = arange(min(x), max(x), 0.1)
    
    pE1 = exponentialPDF(xx, lam1[-1])
    pE2 = exponentialPDF(xx, lam2[-1])
    hist(x, 100, normed=True, histtype='step', label='Data')
    plot(xx, rho[-1] * pE1, label='Population 1')
    plot(xx, (1-rho[-1])*pE2, label='Population 2')
    legend()
    xlabel('Time of Death')
    ylabel('Density')
    xlim((min(x), max(x)))
    yscale('log')
    ylim(1.0 / len(x), 1)
#plotDataAndModel(x,rho,lam1,lam2)

if __name__ == '__main__':
    x = loadDeathData('data1.dat')
    # logLik = logLikelihood(x, 0.8, 0.5, 0.05)
    logLik, rho, lam1, lam2 = EM(x, 0.3, 0.4, 0.2)
    idx = persistors(x, rho[-1], lam1[-1], lam2[-1])
    print(idx)
