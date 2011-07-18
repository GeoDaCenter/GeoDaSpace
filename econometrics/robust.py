import numpy as np
import numpy.linalg as la
from pysal import lag_spatial

def robust_vm(reg,wk=None):
    """
    Robust estimation of the variance-covariance matrix. Estimated by White (default) or HAC (if wk is provided). 
        
    Parameters
    ----------
    
    reg             : Regression object (OLS or TSLS)
                      output instance from a regression model

    wk              : PySAL weights object
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

    >>> wk = pysal.open("examples/kernel_knn15_epanechnikov_3085_random_points.gwt","r").read()
    >>> wk.transform = 'o'
    >>> ols = BaseOLS(y,X, robust='hac', wk=wk)
    >>> ols.vm
    array([[  2.60708153e-01,   6.34129614e-03,  -3.66436005e-02],
           [  5.73679940e-03,   1.69671523e-02,  -3.45021358e-05],
           [ -3.66452363e-02,  -1.17817485e-04,   5.34072367e-03]])

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

    >>> tsls = BaseTSLS(y, X, yd, q=q, robust='hac', wk=wk)
    >>> tsls.vm
    array([[ 0.32274654,  0.03947159, -0.02786886, -0.01724549],
           [ 0.03894785,  0.03572799,  0.00504059, -0.00989053],
           [-0.02793197,  0.00495583,  0.00668599, -0.00268852],
           [-0.01718763, -0.0098797 , -0.00269626,  0.00518021]])

    """
    if hasattr(reg, 'h'): #If reg has H, do 2SLS estimator. OLS otherwise.
        tsls = True
        xu = reg.h * reg.u
    else:
        tsls = False
        xu = reg.x * reg.u
        
    if wk: #If wk do HAC. White otherwise.
        wkxu = lag_spatial(wk,xu)
        psi0 = np.dot(xu.T,wkxu)
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



