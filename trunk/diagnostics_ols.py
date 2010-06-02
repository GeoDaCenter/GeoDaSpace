"""
Diagnostics for OLS estimation. 

Nicholas Malizia <nmalizia@asu.edu>

"""

import pysal
from pysal.common import *
from math import sqrt

db=pysal.open("/Users/Nick/Documents/Academic/Data/GeoDa/columbus/columbus.dbf","r")
y = np.array(db.by_col("CRIME"))
y = np.reshape(y, (49,1))
X = []
X.append(db.by_col("INC"))
X.append(db.by_col("HOVAL"))
X = np.array(X).T
x = X
x = np.hstack((np.ones(y.shape),x))
xt = np.transpose(x)
xtx = np.dot(xt,x)
xtxi = la.inv(xtx)
xty = np.dot(xt,y)
betas = np.dot(xtxi,xty)
predy = np.dot(x,betas)
u = y-predy
utu = np.sum(u**2)
n,k = x.shape
sigmasquare = utu / n


def MultiCollinearity(x):

    """
    Calculates the multicollinearity condition index according to Belsey, Kuh and Welsh (1980)

    Parameters
    ----------

    xtx             : array
                      x by x array of the independent variables and the constant (X'X), imported from OLS.  

    Returns
    -------

    ci_results      : float
                      value for the multicollinearity condition index. 

    References
    ----------
            
    [1] D. Belsley, E. Kuh, and R. Welsch. 1980. Regression Diagnostics. Wiley, New York.

    """

    #get xtx matrix from OLS.py, for now i'm just calculating it on my own. 

    #calculation of condition index
    diag = np.diagonal(xtx)
    scale = xtx/diag    
    eigval = np.linalg.eigvals(scale)
    max_eigval = max(eigval)
    min_eigval = min(eigval)
    ci_results = sqrt(max_eigval/min_eigval)
    return ci_results

def BreuschPagan(parameters):

    #regress residuals using the same independent variables from the original OLS, the only difference is that now the dependent variable is the residuals. 
    usquare = u**2
    usquaremean = np.mean(usquare)
    uscale = usquare/usquaremean

    #which sigmasquare do i use at this point? utu/n or utu/(n-k)?


def White(parameters):



def JarqueBera(parameters):

    """
    Jarque-Bera test for normality in the residuals. 

    Parameters
    ----------
    u               : array
                      residuals from OLS estimation used to test for normality

    Returns
    ------- 
    results         : dictionary
                      contains the statistic (jb) for the Jarque-Bera test and the associated p-value (p-value)

    df              : integer
                      degrees of freedom associated with the test (always 2)

    jb              : float
                      value of the test statistic

    pvalue          : float
                      p-value associated with the statistic (chi^2 distributed with 2 df)


    References
    ----------
            
    [1] C. Jarque and A. Bera. 1980. Efficient tests for normality, homoscedasticity and serial independence of regression residuals. Economics Letters, 6(3):255–259.

    """

    u2 = u**2
    u3 = u**3
    u4 = u**4
    mu2 = np.mean(u2)
    mu3 = np.mean(u3)
    mu4 = np.mean(u4)
    n = len(u)
    S = mu3/(mu2**(1.5))
    K = (mu4/(mu2**2))
    jb = n*(((S**2)/6)+((K-3)**2)/24)
    pvalue=stats.chisqprob(jb,2)
    jb_results={"df":2,"jb":jb,'pvalue':pvalue}
    return jb_results


def KoenkerBassett(parameters):




