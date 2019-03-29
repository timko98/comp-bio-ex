""" Computational Biology I: Quantitative Data Analysis
                Fruehjahrsemester 2019
                       Exercise 5
"""
import numpy as np
from scipy.integrate import quad
from scipy import inf
import matplotlib.pyplot as plt



def gaussian_likelihood(mu,sigma2,x):
    """
    Call :
       lik = gaussian_likelihood(mu,sigma2,x)
    Input argument:
       mu:  float or numpy array (mean fold change)
       sigma2: float (variance)
       x: numpy array (observed fold changes)
    Output argument:
       lik: float if mu is float and numpy array if mu numpy array (likelihood)
    Example:
       In [1]: x = np.array([1,2,4,0])
       
       In [2]: gaussian_likelihood(0.34,np.var(x),x)
       Out[2]: 0.00011634341807197585

       In [3]: gaussian_likelihood(np.arange(0,3,0.5),np.var(x),x)
       Out[3]: 
       array([4.35642886e-05,   1.71685844e-04,   4.28354889e-04,
              6.76609901e-04,   6.76609901e-04,   4.28354889e-04])
    """
    n = len(x)

    lik = ((1 / ((2*np.pi*sigma2)**(n/2))) * (np.exp((-n/(2*sigma2))*(((np.mean(x)-mu)**2) + np.var(x)))))
    return(lik)

def student_likelihood(mu,x):
    """
    Call :
       lik = student_likelihood(mu,x)
    Input argument:
       mu: float or numpy array (mean fold change)
       x: numpy array (observed fold changes)
    Output argument:
       lik: numpy array (likelihood)
    Example:
       In [1]: x = np.array([1,2,4,0])

       In [2]: student_likelihood(0.34,x)
       Out[2]: 0.37917854799162326

       In [3]: student_likelihood(np.arange(0,3,0.5),x)
       Out[3]: 
       array([ 0.26895718,  0.44552819,  0.70945206,  0.95862404,  0.95862404,
               0.70945206])
    """
    n = len(x)
    lik = (1 + ((np.mean(x)-mu)**2/(np.var(x))))**(-(n-1)/2)
    return(lik)

def gaussian_posterior(mu,x):
    """
    Call :
       post = gaussian_posterior(mu,x)
    Input argument:
       mu: float or numpy array (mean fold change)
       x: numpy array (observed fold changes)
    Output argument:
       post: float or numpy array (posterior)
    Example:
       In [1]: x = np.array([1,2,4,0])

       In [2]: gaussian_posterior(0.34,x)
       Out[2]: 0.087609796873199516

       In [3]: gaussian_posterior(np.arange(0,3,0.5),x)
       Out[3]: 
       array([ 0.03280511,  0.12928417,  0.32256302,  0.50950588,  0.50950588,
               0.32256302])
    """
    n = len(x)
    post = (1 / (np.sqrt(np.var(x)) * np.sqrt(2*np.pi/n)))* np.exp((-(n/(2*np.var(x)))) * (np.mean(x)-mu)**2)
    return(post)

def student_posterior(mu,x):
    """
    Call :
       post = student_posterior(mu,x)
    Input argument:
       mu: float or numpy array (mean fold change)
       x: numpy array (observed fold changes)
    Output argument:
       post: numpy array (posterior)
    Example:
       In [1]: x = np.array([1,2,4,0])

       In [2]: student_posterior(0.34,x)
       Out[2]: 0.12818574525476051

       In [3]: student_posterior(np.arange(0,3,0.5),x)
       Out[3]: 
       array([ 0.09092412,  0.15061602,  0.23983857,  0.32407407,  0.32407407,
               0.23983857])
    """

    integral, error = quad(student_likelihood, -inf, inf, args=x) # To be completed! Use -inf and inf as boundary, we imported them from you from scipy

    post = student_likelihood(mu, x)/integral
    return(post)

def expectedFoldChange(x,dist,mu_l=-100,mu_u=100):
    """
    Call :
       meanFC,minFC,maxFC = expectedFoldChange(x,dist)
    Input argument:
       x: numpy array (observed fold changes)
       dist: distribution. 0 = Student-t, 1 = Gaussian approximation 
       mu_l, mu_u are the lower and upper values of mu where we do integrate
    Output argument:
       meanFC: float (mean fold change)
       minFC: float (lower boundary of 95 % probability interval)
       maxFC: float (upper boundary of 95 % probability interval)
    Example:
       In [1]: x = np.array([1,2,4,0])

       In [2]: expectedFoldChange(x,0)
       Out[2]: (1.7494257026761302, -2.7399999999502427, 6.2500000000543565)

       In [3]: expectedFoldChange(x,1)
       Out[3]: (1.7499999999991027, 0.30000000005131255, 3.2000000000527962)
    """
    Dmu = 0.01  # Use this as "Delta mu"

    # TODO Some really far decimals are incorrect

    if dist == 0:
        p = student_posterior(np.arange(mu_l, mu_u+Dmu, Dmu), x)*Dmu
        mean = np.arange(mu_l, mu_u+Dmu, Dmu) * p
        meanFC = np.sum(mean, dtype=np.float64)
        min_index = np.argmax(np.cumsum(p) > 0.025)
        minFC = mu_l + min_index*Dmu
        max_index = np.argmax(np.cumsum(p) > 0.975)
        maxFC = mu_l + max_index*Dmu
    if dist == 1:
        p = gaussian_posterior(np.arange(mu_l, mu_u + Dmu, Dmu), x) * Dmu
        mean = np.arange(mu_l, mu_u + Dmu, Dmu) * p
        meanFC = np.sum(mean)
        min_index = np.argmax(np.cumsum(p) > 0.025)
        minFC = mu_l + min_index * Dmu
        max_index = np.argmax(np.cumsum(p) > 0.975)
        maxFC = mu_l + max_index * Dmu


    return(meanFC,minFC,maxFC)


gene1 = np.array([-0.5989, 0.9163, -1.1192])
gene2 = np.array([ 2.6043,  2.4013, 2.8432, 1.9412, 0.298])
gene3 = np.array([ 2.9973,  4.5676, 1.8934, -1.1978, 1.7192, 4.0529, 0.325, -1.9837, 4.9612, -1.2523])
gene4 = np.array([-2.729, 5.9134, -2.9845])
gene5 = np.array([1.9134, 0.015])

def plot_gaussian_posterior(x):
    mu = np.arange(min(x)-2, max(x)+2,0.01)
    plt.plot(mu,gaussian_posterior(mu,x))
    plt.xlabel('mu')
    plt.ylabel('P(mu|x)')
    return
def plot_student_posterior(x):
    mu = np.arange(min(x)-2, max(x)+2,0.01)
    plt.plot(mu,student_posterior(mu,x))
    plt.xlabel('mu')
    plt.ylabel('P(mu|x)')
    return

def positiveFoldChange(x,dist,mu_u=100):
    """
    Call :
       p_positive = positiveFoldChange(x,dist)
    Input argument:
       x: numpy array (observed fold changes)
       dist: distribution. 0 = Student-t, 1 = Gaussian approximation 
       mu_u is the upper limit of integration
    Output argument:
       p_positive: float ( P(mu>0|x) )
    Example:
       In [1]: x = np.array([1,2,4,0])

       In [2]: positiveFoldChange(x,0)
       Out[2]: 0.88227853320357097

       In [3]: positiveFoldChange(x,1)
       Out[3]: 0.99118291198455866
    """
    Dmu = 0.01 # As before keep this step size for numerical integration

    if dist == 0:
        p_positive = np.sum(student_posterior(np.arange(0, mu_u + Dmu, Dmu), x) * Dmu)
    if dist == 1:
        p_positive = np.sum(gaussian_posterior(np.arange(0, mu_u + Dmu, Dmu), x) * Dmu)

    return(p_positive)


if __name__ == '__main__':
    x = np.array([1, 2, 4, 0])
    r = positiveFoldChange(x,1)
    print(r)
