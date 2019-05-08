""" Computational Biology I: Quantitative Data Analysis
                Fruehjahrsemester 2019
                       Exercise 9
"""

from numpy import sqrt, arange, meshgrid, matrix, array, exp
import matplotlib.pyplot as plt
from matplotlib import cm


def Z(lambda1, lambda2):
    '''
    z = Z(lambda1, lambda2)
    z: float number
    lambda1, lambda2: float numbers

    Example:
    Z(0.1, 0.25)
    =>
    114.28571428571429
    Z(2, 1)
    =>
    0.16666666666666666
    '''


    return(z)

def lambdas(mean_a, mean_b):
    '''
    l = lambdas(mean_a, mean_b)
    l: list of floats
    mean_a, mean_b: float numbers

    Example:
    lambdas(10, 2)
    =>
    [0.11043560762610401,
    0.33956439237389596,
    0.94782196186948009,
    -0.19782196186948001]
    
    lambdas(2.4, 3.1)
    =>
    [0.6643009333724671,
    -2.6881104571819896,
    0.45344443867938028,
    3.0488597087860572]
    '''

    return(result)


def get_means_xyz(mean_a, mean_b):
    '''
    m = get_means_xyz(mean_a, mean_b)
    m: list of floats
    mean_a, mean_b: floats
    
    Example:
    get_means_xyz(3.2, 7.3)
    =>
    [1.3873397496678903, 1.8126602503321101, 5.9126602503321095]
    '''
    

    return(result)


def get_covariance(mean_a, mean_b):
    '''
    covar = get_covariance(mean_a, mean_b)
    covar: matrix of floats
    mean_a, mean_b: floats

    Example:
    get_covariance(10.9, 4.8)
    =>
    matrix([[ 82.1279196 ,   4.32358638],
            [  4.32358638,  11.72566678]])
    '''

    return(cov_mat)

    

def plot_mean():
    a = arange(0.001, 10, 0.1)
    b = arange(0.001, 10, 0.1)

    A, B = meshgrid(a, b)

    means_x = []
    means_y = []
    means_z = []
    for i in range(len(a)):
        tmp_x = []
        tmp_y = []
        tmp_z = []
        for j in range(len(a)):
            m = get_means_xyz(A[i,j],B[i,j])
            tmp_x.append(m[0])
            tmp_y.append(m[1])
            tmp_z.append(m[2])
        means_x.append(tmp_x)
        means_y.append(tmp_y)
        means_z.append(tmp_z)
    
    plt.subplot(131)
    plt.imshow(means_x, interpolation='bilinear', origin='lower', extent=(0.001, 10, 0.001,10))
    plt.xlabel('mean a')
    plt.ylabel('mean b')
    plt.title('Mean x')
    plt.subplot(132)
    plt.imshow(means_y, interpolation='bilinear', origin='lower', extent=(0.001, 10, 0.001,10))
    plt.xlabel('mean a')
    plt.ylabel('mean b')
    plt.title('Mean y')
    plt.subplot(133)
    plt.imshow(means_z, interpolation='bilinear', origin='lower', extent=(0.001, 10, 0.001,10))
    plt.xlabel('mean a')
    plt.ylabel('mean b')
    plt.title('Mean z')
    plt.show()

