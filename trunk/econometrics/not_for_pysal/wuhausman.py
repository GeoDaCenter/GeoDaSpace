"""
Diagnostics for two stage least squares regression estimations. 
        
"""

import pysal
from pysal.common import *
from scipy.stats import pearsonr
from math import sqrt


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
    >>> from twosls import BaseTSLS as TSLS
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
    from pysal.spreg.ols import BaseOLS as OLS
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
