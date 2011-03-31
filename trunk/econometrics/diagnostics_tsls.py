"""
Diagnostics for two stage least squares regression estimations. 
        
"""

import pysal
from pysal.common import *
from math import sqrt
from twosls import BaseTSLS as TSLS
from pysal.spreg.ols import BaseOLS as OLS



def f_stat_tsls(reg):
    """
    Calculates the f-statistic and associated p-value for a two stage least
    squares regression. 
    
    Parameters
    ----------
    reg             : TSLS regression object
                      output instance from a regression model

    Returns
    ----------
    fs_result       : tuple
                      includes value of F statistic and associated p-value

    References
    ----------
    .. [1] J.M. Wooldridge. 1990. A note on the Lagrange multiplier and F-
       statistics for two stage least squares regressions. Economics Letters
       34, 151-155.  

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import diagnostics_tsls as diagnostics
    >>> from twosls import BaseTSLS as TSLS
    >>> db = pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T
    >>> yd = []
    >>> yd.append(db.by_col("HOVAL"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> reg = TSLS(y, X, yd, q)
    >>> testresult = diagnostics.f_stat_tsls(reg)
    >>> print("%12.11f"%testresult[0],"%12.11f"%testresult[1])
    ('7.40058418460', '0.00163476698')

    """ 
    k = reg.k            # (scalar) number of ind. vars (includes constant)
    n = reg.n            # (scalar) number of observations
    utu = reg.utu        # (scalar) residual sum of squares
    predy = reg.predy    # (array) vector of predicted values (n x 1)
    mean_y = reg.mean_y  # (scalar) mean of dependent observations
    Q = utu
    ssr_intercept = OLS(reg.y, np.ones(reg.y.shape), constant=False).utu
    u_2nd_stage = reg.y - np.dot(reg.xp, reg.betas)
    ssr_2nd_stage = np.sum(u_2nd_stage**2)
    U = ssr_intercept - ssr_2nd_stage
    fStat = (U/(k-1))/(Q/(n-k))
    pValue = stats.f.sf(fStat,k-1,n-k)
    fs_result = (fStat, pValue)
    return fs_result


def t_stat(reg, z_stat=False):
    """
    Calculates the t-statistics (or z-statistics) and associated p-values.
    
    Parameters
    ----------
    reg             : regression object
                      output instance from a regression model
    z_stat          : boolean
                      If True run z-stat instead of t-stat
        
    Returns
    -------    
    ts_result       : list of tuples
                      each tuple includes value of t statistic (or z
                      statistic) and associated p-value

    References
    ----------
    .. [1] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper
       Saddle River.

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> import pysal.spreg.diagnostics as diagnostics
    >>> from pysal.spreg.ols import BaseOLS as OLS
    >>> from twosls import BaseTSLS as TSLS
    >>> db = pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> reg = OLS(y,X)
    >>> # t-stat for OLS
    >>> testresult = diagnostics.t_stat(reg)
    >>> print("%12.12f"%testresult[0][0], "%12.12f"%testresult[0][1], "%12.12f"%testresult[1][0], "%12.12f"%testresult[1][1], "%12.12f"%testresult[2][0], "%12.12f"%testresult[2][1])
    ('14.490373143689', '0.000000000000', '-4.780496191297', '0.000018289595', '-2.654408642718', '0.010874504910')
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T    
    >>> yd = []
    >>> yd.append(db.by_col("HOVAL"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> reg = TSLS(y, X, yd, q)
    >>> # z-stat for TSLS
    >>> testresult = diagnostics.t_stat(reg, z_stat=True)
    >>> print("%12.10f"%testresult[0][0], "%12.10f"%testresult[0][1], "%12.10f"%testresult[1][0], "%12.10f"%testresult[1][1], "%12.10f"%testresult[2][0], "%12.10f"%testresult[2][1])
    ('5.8452644705', '0.0000000051', '0.3676015668', '0.7131703463', '-1.9946891308', '0.0460767956')
    """ 
    
    k = reg.k           # (scalar) number of ind. vas (includes constant)
    n = reg.n           # (scalar) number of observations
    vm = reg.vm         # (array) coefficients of variance matrix (k x k)
    betas = reg.betas   # (array) coefficients of the regressors (1 x k) 
    variance = vm.diagonal()
    tStat = betas.reshape(len(betas),)/ np.sqrt(variance)
    ts_result = []
    for t in tStat:
        if z_stat:
            ts_result.append((t, stats.norm.sf(abs(t))*2))
        else:
            ts_result.append((t, stats.t.sf(abs(t),n-k)*2))
    return ts_result



def wuhausman(tslsreg):
    """
    Computes the Wu-Hausman specification test in the form of an F test as 
    formulated by Wu. The Wald statistic advocated by Hausman is avoided 
    due to the fact that it requires a potentially unstable inverse. 

    Parameters
    ----------
    tlsreg              : two stage least squares regression object
                          output instance from a two stage least squares
                          regression model

    Returns
    -------
    wuhausman_result    : dictionary
                          contains the statistic (stat) and the associated
                          p-value (pvalue) for the test. 
    stat                : float
                          scalar value for the F test statistic associated 
                          with the formulation of the test by Wu.
    pvalue              : float
                          p-value associated with the statistic (F-
                          distributed with kstar and n-k-kstar degrees of
                          freedom)
                      
    References
    ----------
    .. [1] D. Wu. 1973. Alternative tests of independence between
       stochastic regressors and disturbances. Econometrica. 41(4):733-750. 
    .. [2] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper
       Saddle River.

    Examples
    --------
    >>> db = pysal.open("examples/greene5_1.csv","r")
    >>> y = []
    >>> y.append(db.by_col('ct'))
    >>> y = np.array(y).T
    >>> X = []
    >>> X.append(db.by_col('tbilrate'))
    >>> X.append(db.by_col('clag'))
    >>> X = np.array(X).T
    >>> yd = []
    >>> yd.append(db.by_col('yt'))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col('ylag'))
    >>> q = np.array(q).T
    >>> tslsreg = TSLS(y, X, yd, q=q)
    >>> result = wuhausman(tslsreg)
    >>> print("%2.6f"%result['stat'])
    28.576893
    >>> print("%2.6f"%result['pvalue'])
    0.000000
    """
    kstar = tslsreg.kstar
    k = tslsreg.k
    n = tslsreg.n
    y = tslsreg.y
    yd = tslsreg.yend
    z = tslsreg.h   #matrix of exogenous x and instruments for endogenous
    x = tslsreg.z   #matrix of exogenous x and endogenous

    # creating predictions for X* in Greene's equation 5-24
    # NOTE - not currently sure how to handle multiple endogenous and
    # instruments, this needs to be discussed
    full = x
    for i in range(kstar):
        part1 = OLS(yd[:,i],z,constant=False)
        ydhat = np.reshape(part1.predy,(n,1))
        full = np.hstack((full,ydhat))

    # although a t-statistic could be used in the case of
    # a single variable, it was simpler to just use an F
    # test for all cases
    ssr_ur = OLS(y,full,constant=False).utu
    ssr_r = OLS(y,x,constant=False).utu

    # calculate the F test and significance
    num = (ssr_r-ssr_ur)/kstar
    den = ssr_ur/(n-k-kstar)
    fstat = num/den
    pvalue = stats.f.sf(fstat,kstar,(n-k-kstar))
    wuhausman_result = {'stat':fstat,'pvalue':pvalue}
    return wuhausman_result 





def _test():
    import doctest
    doctest.testmod()

                     
if __name__ == '__main__':
    _test()    

