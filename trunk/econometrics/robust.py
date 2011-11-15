import numpy as np
import numpy.linalg as la
from pysal import lag_spatial

def robust_vm(reg, gwk=None):
    """
    Robust estimation of the variance-covariance matrix. Estimated by White (default) or HAC (if wk is provided). 
        
    Parameters
    ----------
    
    reg             : Regression object (OLS or TSLS)
                      output instance from a regression model

    gwk             : PySAL weights object
                      Optional. Spatial weights based on kernel functions
                      If provided, returns the HAC variance estimation
                      
    Returns
    --------
    
    psi             : kxk array
                      Robust estimation of the variance-covariance
                      
    Examples
    --------
    
    >>> import numpy as np
    >>> import pysal
    >>> from ols import BaseOLS
    >>> from twosls import BaseTSLS
    >>> db=pysal.open("examples/NAT.dbf","r")
    >>> y = np.array(db.by_col("HR90"))
    >>> y = np.reshape(y, (y.shape[0],1))
    >>> X = []
    >>> X.append(db.by_col("RD90"))
    >>> X.append(db.by_col("DV90"))
    >>> X = np.array(X).T                       

    Example with OLS with unadjusted standard errors

    >>> ols = BaseOLS(y,X)
    >>> ols.vm
    array([[ 0.17004545,  0.00226532, -0.02243898],
           [ 0.00226532,  0.00941319, -0.00031638],
           [-0.02243898, -0.00031638,  0.00313386]])

    Example with OLS and White
    
    >>> ols = BaseOLS(y,X, robust='white')
    >>> ols.vm
    array([[ 0.24491641,  0.01092258, -0.03438619],
           [ 0.01092258,  0.01796867, -0.00071345],
           [-0.03438619, -0.00071345,  0.00501042]])
    
    Example with OLS and HAC

    >>> wk = pysal.kernelW_from_shapefile('examples/NAT.shp',k=15,function='triangular', fixed=False)
    >>> wk.transform = 'o'
    >>> ols = BaseOLS(y,X, robust='hac', gwk=wk)
    >>> ols.vm
    array([[ 0.29213585,  0.01670363, -0.0394821 ],
           [ 0.01655547,  0.02295803, -0.00116872],
           [-0.03941484, -0.00119075,  0.00568315]])

    Example with 2SLS and White

    >>> yd = []
    >>> yd.append(db.by_col("UE90"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("UE80"))
    >>> q = np.array(q).T
    >>> tsls = BaseTSLS(y, X, yd, q=q, robust='white')
    >>> tsls.vm
    array([[ 0.29569954,  0.04119843, -0.02496858, -0.01640185],
           [ 0.04119843,  0.03647762,  0.004702  , -0.00987345],
           [-0.02496858,  0.004702  ,  0.00648262, -0.00292891],
           [-0.01640185, -0.00987345, -0.00292891,  0.0053322 ]])

    Example with 2SLS and HAC

    >>> tsls = BaseTSLS(y, X, yd, q=q, robust='hac', gwk=wk)
    >>> tsls.vm
    array([[ 0.41985248,  0.06823097, -0.02883901, -0.02788095],
           [ 0.06867058,  0.04887499,  0.00497445, -0.01367751],
           [-0.02856442,  0.00501408,  0.00721951, -0.00321607],
           [-0.02810132, -0.01364912, -0.00318198,  0.00713251]])

    """
    if hasattr(reg, 'h'): #If reg has H, do 2SLS estimator. OLS otherwise.
        tsls = True
        xu = reg.h * reg.u
    else:
        tsls = False
        xu = reg.x * reg.u
        
    if gwk: #If gwk do HAC. White otherwise.
        gwkxu = lag_spatial(gwk,xu)
        psi0 = np.dot(xu.T,gwkxu)
    else:
        psi0 = np.dot(xu.T,xu)
        
    if tsls:
        psi1 = np.dot(reg.varb,reg.zthhthi)
        psi = np.dot(psi1,np.dot(psi0,psi1.T))
    else:
        psi = np.dot(reg.xtxi,np.dot(psi0,reg.xtxi))
        
    return psi
    
def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()



