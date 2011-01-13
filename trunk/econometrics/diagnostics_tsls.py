"""
Diagnostics for two stage least squares regression estimations. 
        
"""

import pysal
from pysal.common import *
from math import sqrt


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
    >>> import diagnostics
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
    import ols as OLS
    ssr_intercept = OLS.BaseOLS(reg.y, np.ones(reg.y.shape), constant=False).utu
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
    >>> import diagnostics
    >>> from ols import BaseOLS as OLS
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
    >>> print("%12.12f"%testresult[0][0], "%12.12f"%testresult[0][1], "%12.12f"%testresult[1][0], "%12.12f"%testresult[1][1], "%12.12f"%testresult[2][0], "%12.12f"%testresult[2][1])
    ('5.845264470459', '0.000000005058', '0.367601566836', '0.713170346347', '-1.994689130783', '0.046076795581')
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



def hausman(olsreg,tslsreg):
    """
    Computes the Hausman specification test in the form of a Wald statistic.

    Parameters
    ----------
    olsreg          : ordinary least squares regression object
                      output instance from an ordinary least squares
                      regression model
    tlsreg          : two stage least squares regression object
                      output instance from a two stage least squares
                      regression model

    Returns
    -------
    hausman_result  : dictionary
                      contains the statistic (hausman), degrees of freedom
                     (df) and the associated p-value (pvalue) for the test. 
    hausman         : float
                      scalar value for the Hausman test statistic.
    df              : integer
                      degrees of freedom associated with the test
    pvalue          : float
                      p-value associated with the statistic (chi^2
                      distributed with kstar degrees of freedom)
                      
    References
    ----------
    .. [1] W. Greene. 2003. Econometric Analysis. Prentice Hall, Upper
       Saddle River.

    Examples
    --------

    """
    b_iv = tlsreg.delta
    b_ls = olsreg.betas
    v_iv = tlsreg.xptxpi
    v_ls = olsreg.xtxi
    sig2 = olsreg.sig2n_k   # Greene specifies this is the correct sig2 to use, Stata gives an option     
    df = tlsreg.kstar       # degrees of freedom specified by Greene, Stata uses tlsreg.k 

    d = b_iv-b_ls
    dt = d.T
    part1 = la.pinv(v_iv-v_ls)
    part2 = np.dot(dt,part1)
    part3 = np.dot(part2,d)
    hausman = part3/sig2
    pvalue=stats.chisqprob(hausman,df)
    hausman_result = {'hausman':hausman,'df':df,'pvalue':pvalue}
    return hausman_result

