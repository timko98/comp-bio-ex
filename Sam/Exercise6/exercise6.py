""" Computational Biology I: Quantitative Data Analysis
                Fruehjahrsemester 2019
                       Exercise 6
"""

import numpy as np
import matplotlib.pyplot as plt

"""
Predefined Functions :
"""

def exponentialPDF(x,a):
    p = a*np.exp(-a*x)
    p[x<0] = 0.0
    return(p)

def normalPDF(x,mu,sigma):
    p = 1.0/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2.0*sigma**2.0))
    return(p)

def loadData(file_name):
    """
    Call:
       length,count = loadData(file_name)
    Input argument:
       file_name: string (bed file name)
    Output argument:
       length: numpy array
       count: numpy array
    Example:
       length,count = loadData('H3K36ac.bed')
       =>
       length: array([40, 27, 81, ..., 31, 28, 26])
       count: array([  200,  9666, 14823, ...,  3131,  1036,  1716])
    """
    length = []
    count = []
    for line in open(file_name):
        elm = line.split()
        length.append(int(elm[2])-int(elm[1]))
        count.append(int(elm[4]))
    return(np.array(length),np.array(count))

def plotData(x,rho,mu,sigma,lam):
    """
    Call:
       plotData(x,rho,mu,sigma,lam)
    Input argument:
       x: numpy array
       rho: float
       mu: float
       sigma: float
       lam: float
    """
    xx = np.arange(min(x),max(x),0.01)
    pG = normalPDF(xx,mu,sigma)
    pE = exponentialPDF(xx,lam)
    mix = rho*pG + (1-rho)*pE
    plt.hist(x,100,normed=True,histtype='step',label='Data')
    plt.plot(xx,rho*pG,label='Foreground')
    plt.plot(xx,(1-rho)*pE,label='Background')
    plt.plot(xx,mix,'--k',label='Mixture')
    plt.legend()
    plt.xlabel('Log[Read Count]')
    plt.ylabel('Density')
    plt.xlim((min(x),max(x)))
    plt.show()


"""
Functions to write :
"""

def transformData(length,count,pseudoCount=0.5):
    """
    Call:
       x = transformData(length,count,pseudoCount)
    Input argument:
       length: numpy array
       count: numpy array
       pseudoCount: float (optional, default=0.5)
    Output argument:
       x: numpy array
    Example:
       length,count = loadData('H3K36ac.bed')
       x = transformData(length,count)
       =>
       x: array([ 1.70474809,  5.88192866,  5.21221467, ...,  4.6200588 ,
                  3.62434093,  4.19720195])
    """
    x = np.log(count / length + pseudoCount)
    return(x)

def mixture_logLikelihood(x,rho,mu,sigma,lam):
    """
    Call:
       logLik = mixture_logLikelihood(x,rho,mu,sigma,lam)
    Input argument:
       x: numpy array
       rho: float
       mu: float
       sigma: float
       lam: float
    Output argument:
       logLik: float
    Example:
       length,count = loadData('H3K36ac.bed')
       x = transformData(length,count)
       logLik = mixture_logLikelihood(x,0.5,1.3,0.4,0.3)
       =>
       logLik: -7819.23354303
    """
    logLik = np.sum(np.log(rho*normalPDF(x,mu,sigma) + (1-rho)*exponentialPDF(x, lam)))
    return(logLik)

def EM(x,rho,mu,sigma,lam,eps=0.0001):
    """
    Call:
       logLik,rho,mu,sigma,lam = EM(x,rho,mu,sigma,lam)
    Input argument:
       x: numpy array
       rho: float
       mu: float
       sigma: float
       lam: float
       eps: float, stopping criterion L[j-L[j-1] < eps (optional, default=0.5)
    Output argument:
       logLik: numpy array
       rho: float
       mu: float
       sigma: float
       lam: float
    Example:
       length,count = loadData('H3K36ac.bed')
       x = transformData(length,count)
       logLik,rho,mu,sigma,lam = EM(x,0.5,1.3,0.4,0.3)
       =>
       logLik :
       array([-7819.23354303, -6334.35454986, -6249.45611846, -6229.7748527 ,
              -6223.97055711, -6220.90287621, -6216.07997837, -6203.08066084,
              -6164.76931903, -6060.00462573, -5841.32583076, -5557.95004609,
              -5315.00841266, -5121.89701104, -4965.1730112 , -4848.09686201,
              -4770.94687666, -4724.77890576, -4698.12025621, -4682.7029945 ,
              -4673.65862314, -4668.24624408, -4664.93395142, -4662.86065095,
              -4661.53507748, -4660.67120558, -4660.09868606, -4659.71370799,
              -4659.45160301, -4659.27125927, -4659.1460564 , -4659.05847392,
              -4658.996814  , -4658.95316818, -4658.9221315 , -4658.89997527,
              -4658.88410636, -4658.87270878, -4658.8645032 , -4658.85858375,
              -4658.85430616, -4658.85121052, -4658.84896747, -4658.84734046,
              -4658.84615924, -4658.845301  , -4658.84467703, -4658.84422312,
              -4658.84389277, -4658.84365224, -4658.84347706, -4658.84334943,
              -4658.84325641])
       rho :   0.8773466973309262
       mu :    4.7786352834847694
       sigma : 1.2833068565940386
       lam :   0.5107133621362869
    """
    n = len(x)
    error = eps + 1
    logLik = [mixture_logLikelihood(x, rho, mu, sigma, lam)]
    while error > eps:
        L1 = normalPDF(x, mu, sigma)
        L2 = exponentialPDF(x, lam)

        p1 = (rho * L1) / (rho * L1 + (1-rho) * L2)
        p2 = ((1-rho) * L2) / (rho * L1 + (1-rho) * L2)

        rho = np.sum(p1) / n
        sigma = np.sqrt(np.sum((x-mu)**2 * p1) / np.sum(p1))
        mu = np.sum(x * p1) / np.sum(p1)
        lam = np.sum(p2)  / np.sum(x * p2)

        logLik.append(mixture_logLikelihood(x, rho, mu, sigma, lam))
        error = np.abs(logLik[-1] - logLik[-2])

    return(logLik,rho,mu,sigma,lam)

def probForeground(x,rho,mu,sigma,lam):
    """
    Call:
       p = probForeground(x,rho,mu,sigma,lam)
    Input argument:
       x: numpy array
       rho: float
       mu: float
       sigma: float
       lam: float
    Output argument:
       p: numpy array
    Example:
       length,count = loadData('H3K36ac.bed')
       x = transformData(length,count)
       logLik,rho,mu,sigma,lam = EM(x,0.5,1.3,0.4,0.3)
       p = probForeground(x,rho,mu,sigma,lam)
       =>
       p : array([ 0.37122812,  0.98378583,  0.98330781, ...,  0.9786056 ,
                   0.94870896,  0.97102834])
    """
    L1 = normalPDF(x, mu, sigma)
    L2 = exponentialPDF(x, lam)

    p1 = (rho * L1) / (rho * L1 + (1 - rho) * L2)
    p2 = ((1 - rho) * L2) / (rho * L1 + (1 - rho) * L2)

    p = p1 / (p1 + p2)

    return(p)

if __name__ == "__main__":
    length, count = loadData('H3K36ac.bed')
    x = transformData(length,count)
    logLik,rho,mu,sigma,lam = EM(x,0.5,1.3,0.4,0.3)

    plt.semilogy(length, count, '.')
    plt.xlabel("length")
    plt.ylabel("counts")
    plt.show()

    print(logLik)
    print(rho, mu, sigma, lam)
    print(probForeground(x,rho,mu,sigma,lam))
    plotData(x, rho, mu, sigma, lam)

