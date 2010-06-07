"""
Diagnostics for OLS estimation. 

To Do List:

    * Resolve conflict between GeoDa and R in the Breusch-Pagan test.
    * Ask questions regarding White/Breusch/Koenker tests & auxiliary regressions. 
    * Complete White and Koenker-Bassett diagnostics.
    * Import other diagnostic tests from OLS class to consolidate diagnostics.
    * Will these diagnostics be used for the 2SLS regression as well? 
    
"""

import pysal
from pysal.common import *
from math import sqrt


def LogLikelihood(ols):

    """
    Calculates the log-likelihood value for the regression. 
    
    Parameters
    ----------

    ols             : OLS_dev
                      instance from an OLS_dev regression

    Returns
    -------

    ll_result      : float
                     value for the log-likelihood of the regression.

    References
    ----------

    [1] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper Saddle River.

    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> from econometrics.ols import OLS_dev as OLS
    >>> from econometrics import diagnostics as diagnostics
    >>> db = pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> ols = OLS(X,y)
    >>> testresult = diagnostics.LogLikelihood(ols)
    >>> testresult
    -187.3772388121491

    """

    n = ols.n       # (scalar) number of observations
    utu = ols.utu   # (scalar) residual sum of squares
    ll_result = -0.5*(n*(np.log(2*math.pi))+n*np.log(utu/n)+(utu/(utu/n)))
    return ll_result   


def AkaikeCriterion(ols):

    """
    Calculates the Akaike Information Criterion

    Parameters
    ----------

    ols             : OLS_dev
                      instance from an OLS_dev regression

    Returns
    -------

    aic_result      : scalar
                      value for Akaike Information Criterion of the regression. 

    References
    ----------

    [1] H. Akaike. 1974. A new look at the statistical identification model. IEEE Transactions on Automatic Control, 19(6):716-723.

    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> from econometrics.ols import OLS_dev as OLS
    >>> from econometrics import diagnostics as diagnostics
    >>> db = pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> ols = OLS(X,y)
    >>> testresult = diagnostics.AkaikeCriterion(ols)
    >>> testresult
    380.7544776242982

    """

    n = ols.n       # (scalar) number of observations
    k = ols.k       # (scalar) number of independent variables (including constant)
    utu = ols.utu   # (scalar) residual sum of squares
    aic_result = 2*k + n*(np.log((2*np.pi*utu)/n)+1)
    return aic_result


def SchwarzCriterion(ols):

    """
    Calculates the Schwarz Information Criterion

    Parameters
    ----------

    ols             : OLS_dev
                      instance from an OLS_dev regression

    Returns
    -------

    bic_result      : scalar
                      value for Schwarz (Bayesian) Information Criterion of the regression. 

    References
    ----------

    [1] G. Schwarz. 1978. Estimating the dimension of a model. The Annals of Statistics, pages 461-464. 

    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> from econometrics.ols import OLS_dev as OLS
    >>> from econometrics import diagnostics as diagnostics
    >>> db = pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> ols = OLS(X,y)
    >>> testresult = diagnostics.SchwarzCriterion(ols)
    >>> testresult
    386.42993851863008

    """

    n = ols.n       # (scalar) number of observations
    k = ols.k       # (scalar) number of independent variables (including constant)
    utu = ols.utu   # (scalar) residual sum of squares
    sc_result = k*np.log(n) + n*(np.log((2*np.pi*utu)/n)+1)
    return sc_result


def MultiCollinearity(ols):

    """
    Calculates the multicollinearity condition index according to Belsey, Kuh and Welsh (1980)

    Parameters
    ----------

    ols             : OLS_dev
                      instance from an OLS_dev regression 

    Returns
    -------

    ci_result       : float
                      scalar value for the multicollinearity condition index. 

    References
    ----------
            
    [1] D. Belsley, E. Kuh, and R. Welsch. 1980. Regression Diagnostics. Wiley, New York.

    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> from econometrics.ols import OLS_dev as OLS
    >>> from econometrics import diagnostics as diagnostics
    >>> db = pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> ols = OLS(X,y)
    >>> testresult = diagnostics.MultiCollinearity(ols)
    >>> testresult
    6.5418277514438046

    """

    xtx = ols.xtx   # (array) k x k projection matrix (includes constant)
    diag = np.diagonal(xtx)
    scale = xtx/diag    
    eigval = np.linalg.eigvals(scale)
    max_eigval = max(eigval)
    min_eigval = min(eigval)
    ci_result = sqrt(max_eigval/min_eigval)
    return ci_result


def JarqueBera(ols):

    """
    Jarque-Bera test for normality in the residuals. 

    Parameters
    ----------

    ols             : OLS_dev
                      instance from an OLS_dev regression 

    Returns
    ------- 
    jb_result       : dictionary
                      contains the statistic (jb) for the Jarque-Bera test and the associated p-value (p-value)

    df              : integer
                      degrees of freedom associated with the test (always 2)

    jb              : float
                      value of the test statistic

    pvalue          : float
                      p-value associated with the statistic (chi^2 distributed with 2 df)


    References
    ----------
            
    [1] C. Jarque and A. Bera. 1980. Efficient tests for normality, homoscedasticity and serial independence of regression residuals. Economics Letters, 6(3):255-259.

    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> from econometrics.ols import OLS_dev as OLS
    >>> from econometrics import diagnostics as diagnostics
    >>> db = pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> ols = OLS(X,y)
    >>> testresult = diagnostics.JarqueBera(ols)
    >>> testresult['df']
    2
    >>> testresult['jb']
    1.835752520075947
    >>> testresult['pvalue']
    0.39936629124876566

    """
    
    n = ols.n               # (scalar) number of observations
    u = ols.u               # (array) residuals from ols estimation. 
    u2 = u**2                          
    u3 = u**3                          
    u4 = u**4                           
    mu2 = np.mean(u2)       
    mu3 = np.mean(u3)       
    mu4 = np.mean(u4)         
    S = mu3/(mu2**(1.5))    # skewness measure
    K = (mu4/(mu2**2))      # kurtosis measure
    jb = n*(((S**2)/6)+((K-3)**2)/24)
    pvalue=stats.chisqprob(jb,2)
    jb_result={"df":2,"jb":jb,'pvalue':pvalue}
    return jb_result 


def BreuschPagan(ols):

    """
    Calculates the Breusch-Pagan test statistic to check for heteroskedasticity. 

    Parameters
    ----------

    ols             : OLS_dev
                      instance from an OLS_dev regression 

    Returns
    -------

    bp_result       : dictionary
                      contains the statistic (bp) for the Breusch-Pagan test and the associated p-value (p-value)

    bp              : float
                      scalar value for the Breusch-Pagan test statistic.

    df              : integer
                      degrees of freedom associated with the test (k)

    pvalue          : float
                      p-value associated with the statistic (chi^2 distributed with k df)

    References
    ----------
    
    [1] T. Breusch and A. Pagan. 1979. A simple test for heteroscedasticity and random coefficient variation. Econometrica: Journal of the Econometric Society, 47(5):1287-1294.

    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> from econometrics.ols import OLS_dev as OLS
    >>> from econometrics import diagnostics as diagnostics
    >>> db = pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> ols = OLS(X,y)
    >>> testresult = diagnostics.BreuschPagan(ols)
    >>> testresult['df']
    2
    >>> testresult['bp']
    7.2165644721877591
    >>> testresult['pvalue']
    0.027098355486469678
    
    """
 
    k = ols.k           # (scalar) number of independent variables (including constant)
    n = ols.n           # (scalar) number of observations in the regression
    u = ols.u           # (array) residuals from the regression
    x = ols.x           # (array) independent variables in the regression
    xt = ols.xt         # (array) transposed vector of independent variables
    xtxi = ols.xtxi     # (array) k x k projection matrix (includes constant)
    e = u**2            # (array) squared residuals become dependent for auxiliary regression
    xte = np.dot(xt, e)
    g = np.dot(xtxi, xte)   # SHOULD I CONDUCT THIS REGRESSION USING THE OLS CLASS????
    prede = np.dot(x, g)
    q = e - prede
    qtq = np.dot(np.transpose(q), q)
    part1 = e - np.mean(e)
    part2 = part1**2
    part3 = sum(part2)
    r2 = 1 - (qtq/part3) 
    bp_array = n*r2
    bp = bp_array[0,0]
    df = k-1
    pvalue=stats.chisqprob(bp,df)
    bp_result={'df':df,'bp':bp, 'pvalue':pvalue}
    return bp_result 

#def KoenkerBassett(parameters):


#def White(parameters):



def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()

