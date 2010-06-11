"""
Diagnostics for regression estimations. 

To Do List:

    * Resolve conflict between GeoDa and R in the Breusch-Pagan test.
    * Complete White and Koenker-Bassett diagnostics.
    * Add variance inflation factor (more complicated than originally expected). 

        
"""

import pysal
from pysal.common import *
from math import sqrt

def f_stat(reg):

    """
    Calculates the f-statistic and associated p-value of the regression. 
    
    Parameters
    ----------

    reg             : regression object
                      output instance from a regression model
        
    Returns
    ----------

    fs_result       : tuple
                      includes value of F statistic and associated p-value

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
    >>> reg = OLS(X,y)
    >>> testresult = diagnostics.f_stat(reg)
    >>> testresult
    (28.385629224694853, 9.3407471005108332e-09)
    
    """ 
    
    k = reg.k                   # (scalar) number of independent variables (includes constant)
    n = reg.n                   # (scalar) number of observations
    utu = reg.utu               # (scalar) residual sum of squares
    predy = reg.predy           # (array) vector of predicted values (n x 1)
    mean_y = reg.mean_y         # (scalar) mean of dependent observations
    U = np.sum((predy-mean_y)**2)
    Q = utu
    fStat = (U/(k-1))/(Q/(n-k))
    pValue = stats.f.sf(fStat,k-1,n-k)
    fs_result = (fStat, pValue)
    return fs_result


def t_stat(reg):
    """
    Calculates the t-statistics and associated p-values.
    
    Parameters
    ----------

    reg             : regression object
                      output instance from a regression model
        
    Returns
    -------    

    ts_result       : list of tuples
                      each tuple includes value of t statistic and associated p-value

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
    >>> reg = OLS(X,y)
    >>> testresult = diagnostics.t_stat(reg)
    >>> testresult
    [(14.490373143689094, 9.2108899889173982e-19), (-4.7804961912965762, 1.8289595070843232e-05), (-2.6544086427176916, 0.010874504909754612)]
    
    """ 
    
    k = reg.k                   # (scalar) number of independent variables (includes constant)
    n = reg.n                   # (scalar) number of observations
    vm = reg.vm                 # (array) coefficients of variance matrix (k x k)
    betas = reg.betas           # (array) coefficients of the regressors (1 x k) 
    variance = vm.diagonal()
    tStat = betas.reshape(len(betas),)/ np.sqrt(variance)
    rs = {}
    for i in range(len(betas)):
        rs[i] = (tStat[i],stats.t.sf(abs(tStat[i]),n-k)*2)
    ts_result  = rs.values()
    return ts_result

def r2(reg):

    """
    Calculates the R^2 value for the regression. 
    
    Parameters
    ----------

    reg             : regression object
                      output instance from a regression model
        
    Returns
    ----------

    r2_result       : float
                      value of the coefficient of determination for the regression 

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
    >>> reg = OLS(X,y)
    >>> testresult = diagnostics.r2(reg)
    >>> testresult
    0.55240404083742334
    
    """ 

    y = reg.y                   # (array) vector of dependent observations (n x 1)
    mean_y = reg.mean_y         # (scalar) mean of dependent observations
    utu = reg.utu               # (scalar) residual sum of squares
    ss_tot = sum((y-mean_y)**2)
    r2 = 1-utu/ss_tot
    r2_result = r2[0]
    return r2_result


def ar2(reg):

    """
    Calculates the adjusted R^2 value for the regression. 
    
    Parameters
    ----------

    reg             : regression object
                      output instance from a regression model
        
    Returns
    ----------

    ar2_result      : float
                      value of R^2 adjusted for the number of explanatory variables.

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
    >>> reg = OLS(X,y)
    >>> testresult = diagnostics.ar2(reg)
    >>> testresult
    0.5329433469607896

    """ 

    k = reg.k                   # (scalar) number of independent variables (includes constant)
    n = reg.n                   # (scalar) number of observations
    ar2_result =  1-(1-r2(reg))*(n-1)/(n-k)
    return ar2_result


def se_betas(reg):

    """
    Calculates the standard error of the regression coefficients.
    
    Parameters
    ----------

    reg             : regression object
                      output instance from a regression model
        
    Returns
    ----------

    se_result       : array
                      includes standard errors of each regression coefficient (1 x k)
    
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
    >>> reg = OLS(X,y)
    >>> testresult = diagnostics.se_betas(reg)
    >>> testresult
    array([ 4.73548613,  0.33413076,  0.10319868])
    
    """ 

    vm = reg.vm                 # (array) coefficients of variance matrix (k x k)  
    variance = vm.diagonal()
    se_result = np.sqrt(variance)
    return se_result


def log_likelihood(reg):

    """
    Calculates the log-likelihood value for the regression. 
    
    Parameters
    ----------

    reg             : regression object
                      output instance from a regression model

    Returns
    -------

    ll_result       : float
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
    >>> reg = OLS(X,y)
    >>> testresult = diagnostics.log_likelihood(reg)
    >>> testresult
    -187.3772388121491

    """

    n = reg.n       # (scalar) number of observations
    utu = reg.utu   # (scalar) residual sum of squares
    ll_result = -0.5*(n*(np.log(2*math.pi))+n*np.log(utu/n)+(utu/(utu/n)))
    return ll_result   


def akaike(reg):

    """
    Calculates the Akaike Information Criterion

    Parameters
    ----------

    reg             : regression object
                      output instance from a regression model

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
    >>> reg = OLS(X,y)
    >>> testresult = diagnostics.akaike(reg)
    >>> testresult
    380.7544776242982

    """

    n = reg.n       # (scalar) number of observations
    k = reg.k       # (scalar) number of independent variables (including constant)
    utu = reg.utu   # (scalar) residual sum of squares
    aic_result = 2*k + n*(np.log((2*np.pi*utu)/n)+1)
    return aic_result


def schwarz(reg):

    """
    Calculates the Schwarz Information Criterion

    Parameters
    ----------

    reg             : regression object
                      output instance from a regression model

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
    >>> reg = OLS(X,y)
    >>> testresult = diagnostics.schwarz(reg)
    >>> testresult
    386.42993851863008

    """

    n = reg.n       # (scalar) number of observations
    k = reg.k       # (scalar) number of independent variables (including constant)
    utu = reg.utu   # (scalar) residual sum of squares
    sc_result = k*np.log(n) + n*(np.log((2*np.pi*utu)/n)+1)
    return sc_result


def condition_index(reg):

    """
    Calculates the multicollinearity condition index according to Belsey, Kuh and Welsh (1980)

    Parameters
    ----------

    reg             : regression object
                      output instance from a regression model

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
    >>> reg = OLS(X,y)
    >>> testresult = diagnostics.condition_index(reg)
    >>> testresult
    6.5418277514438046

    """

    xtx = reg.xtx   # (array) k x k projection matrix (includes constant)
    diag = np.diagonal(xtx)
    scale = xtx/diag    
    eigval = np.linalg.eigvals(scale)
    max_eigval = max(eigval)
    min_eigval = min(eigval)
    ci_result = sqrt(max_eigval/min_eigval)
    return ci_result


def jarque_bera(reg):

    """
    Jarque-Bera test for normality in the residuals. 

    Parameters
    ----------

    reg             : regression object
                      output instance from a regression model

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
    >>> reg = OLS(X,y)
    >>> testresult = diagnostics.jarque_bera(reg)
    >>> testresult['df']
    2
    >>> testresult['jb']
    1.835752520075947
    >>> testresult['pvalue']
    0.39936629124876566

    """
    
    n = reg.n               # (scalar) number of observations
    u = reg.u               # (array) residuals from regression 
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


def breusch_pagan(reg):

    """
    Calculates the Breusch-Pagan test statistic to check for heteroskedasticity. 

    Parameters
    ----------

    reg             : regression object
                      output instance from a regression model 

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
    >>> reg = OLS(X,y)
    >>> testresult = diagnostics.breusch_pagan(reg)
    >>> testresult['df']
    2
    >>> testresult['bp']
    7.2165644721877591
    >>> testresult['pvalue']
    0.027098355486469678
    
    """
 
    k = reg.k           # (scalar) number of independent variables (including constant)
    n = reg.n           # (scalar) number of observations in the regression
    u = reg.u           # (array) residuals from the regression
    x = reg.x           # (array) independent variables in the regression
    xt = reg.xt         # (array) transposed vector of independent variables
    xtxi = reg.xtxi     # (array) k x k projection matrix (includes constant)
    e = u**2            # (array) squared residuals become dependent for auxiliary regression
    xte = np.dot(xt, e)
    g = np.dot(xtxi, xte)   
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


#def koenker_bassett(parameters):



#def white(parameters):



#def variance_inflation(reg)





def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()

