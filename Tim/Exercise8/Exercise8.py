""" Computational Biology I: Quantitative Data Analysis
                Fruehjahrsemester 2019
                       Exercise 8
"""

import numpy as np
import pylab as py


def loadData(fname):
    """
    Call:
       rowNames,colNames,E = loadData('dataSet8.dat')
    Input argument:
       fname: string (file name)
    Output arguments:
       rowNames: numpy string array
       colNames: numpy string array
       E: numpy array 
 
    """
    header = open(fname).readline()
    colNames = np.array(header.split())
    rowNames = py.loadtxt(fname, skiprows=1, usecols=(0,), dtype=str)
    E = py.loadtxt(fname, skiprows=1, usecols=range(1, len(colNames) + 1), dtype=float)
    return (rowNames, colNames, E)


def Covariance(E):
    """
    Call:
       C = Covariance(E)
    Input argument:
       F: numpy array 
    Output arguments:
       C: numpy array (sample covariance matrix) 
    Example:
       rn,cn,E = loadData('dataSet8.dat')
       C = Covariance(E)
    =>
    C:
    array([[ 12168686.89961562,   7479257.60726226,   5330215.17308303,
         10866184.28290914,   8952466.26875141,   7747397.90935381,
         16291990.40252122],
       [  7479257.60726226,   5947937.28777421,   3623042.04902989,
          7238179.82317247,   6116639.50416963,   5209482.02887227,
         11015277.50521385],
       [  5330215.17308303,   3623042.04902989,   2943146.07043651,
          5068053.24838228,   4123462.68420126,   3756702.45890393,
          7048209.68729213],
       [ 10866184.28290914,   7238179.82317247,   5068053.24838228,
         10367639.40656846,   8149238.46732902,   7421696.90143007,
         14743731.68314564],
       [  8952466.26875141,   6116639.50416963,   4123462.68420126,
          8149238.46732902,   7729612.35613021,   5879880.36944011,
         13135562.54370136],
       [  7747397.90935381,   5209482.02887227,   3756702.45890393,
          7421696.90143007,   5879880.36944011,   5462179.66443427,
         10411877.39044476],
       [ 16291990.40252122,  11015277.50521385,   7048209.68729213,
         14743731.68314564,  13135562.54370136,  10411877.39044476,
         24798866.42069558]])

    """
    G = E.shape[0]
    E_mean = E - (1 / G) * np.sum(E, axis=0)
    C = (1 / G) * np.dot(E_mean.T, E_mean)
    return (C)


def PCA(E):
    """
    Call:
       d,V = PCA(E)
    Input argument:
       E: numpy array 
    Output arguments:
       d: numpy array (1-d vector)
       V: numpy array (2-d matrix; column-vectors form basis of sample space)
    Example:
       rn,cn,E = loadData('dataSet8.dat')
       d,V = PCA(E)
       =>
       d:
       array([65106217.80618648,  2111594.4114361 ,   981997.68629382,
          67832.3145902 ,   243908.07938056,   293515.33696173,
         613001.9953237 ])
       V:
       array([[-0.42217688,  0.22574011, -0.5986572 ,  0.10998145,  0.52009021,
        -0.35968773,  0.02127525],
       ...
       [-0.6042583 , -0.67012803, -0.08268436,  0.03673943, -0.0608554 ,
         0.33481758, -0.24861145]])  
    """
    C = Covariance(E)
    d, V = np.linalg.eig(C)
    return (d, V)


def FOV(E):
    """
    Call:
       fov,TwoComp = FOV(E)
    Input argument:
        E: numpy array 
    Output arguments:
       fov: list (fraction of explained variance for first 2 components) 
       TwoComp : list of np.array (list containing the 2 principal components) 
    Example:
     rn,cn,E = loadData('dataSet8.dat')
     fov,TwoComp = FOV(E)
     =>
     fov:
     [0.9378857699272524, 0.0304185132707194]
     TwoComp:,
     [array([-0.42217688,  0.22574011, -0.5986572 ,  0.10998145,  0.52009021,
        -0.35968773,  0.02127525]),
      array([-0.28408722,  0.07507864,  0.75645989,  0.05071601,  0.38141332,
        -0.32873977, -0.29207536])]
    """
    d, V = PCA(E)
    C = Covariance(E)
    D = np.dot(V.T, np.dot(C, V))
    D_diag = np.diagonal(D)
    frac = np.zeros(2)
    frac[0] = D_diag[0] / np.sum(D_diag)
    frac[1] = D_diag[1] / np.sum(D_diag)
    TwoCom = np.array([V[0, :], V[1, :]])
    return (frac, TwoCom)


def Project2(E):
    """
    Call:
       p1,p2 = Project2(E)
    Input argument:
       E: numpy array 
    Output arguments:
       p1: numpy array (1-d matrix; the projection on the first principal component) 
       p2: numpy array (1-d matrix; the projection on the second principal component) 

    Example:
       rn,cn,E = loadData('dataSet8.dat')
       Project2(E)
       =>
       (array([ -45.87517187, -155.25914581,  -18.93282485, ...,  -11.96916405,
          -1.59295432,   -8.96698298]),
        array([ 25.8301278 , 134.89895448,   9.70570268, ...,   2.6057352 ,
          0.18522836,   5.57246455])) 
    """
    fov, TwoComp = FOV(E)
    p1 = np.dot(TwoComp[0, :], E.T)
    p2 = np.dot(TwoComp[1, :], E.T)
    return (p1, p2)
