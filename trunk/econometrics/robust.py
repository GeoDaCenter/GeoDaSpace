import numpy as np
import numpy.linalg as la
from pysal import lag_spatial

def gls4tsls(y, z, h, u):
    """
    Feasible generalized least squares for the 2SLS model.  This is intended
    to be called from a 2SLS class, but it can be run manually as seen in the
    example below.

    NOTE: no consistency checks
    
    Parameters
    ----------

    y           : array
                  nx1 array of dependent variable
    z           : array
                  nxk array of independent variables, including endogenous
                  variables (assumed to be aligned with y)
    h           : array
                  nxl array of instruments; typically this includes all
                  exogenous variables from x and instruments; (it is different
                  from h seen elsewhere in the library because it can contain
                  a constant)

    Attributes
    ----------

    results     : tuple
                  first element is a kx1 array of estimated coefficients
                  (i.e. betas) and the second is a kxk array for the
                  variance-covariance matrix (i.e [(Z'X)' omega^-1 (Z'X)]^-1 )

    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> from twosls_sp import BaseGM_Lag
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))
    >>> # no non-spatial endogenous variables
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("HOVAL"))
    >>> X = np.array(X).T
    >>> # run gls4tsls manually
    >>> reg=BaseGM_Lag(y, X, w, w_lags=2)
    >>> reg.betas
    array([[ 45.45909249],
           [ -1.0410089 ],
           [ -0.25953844],
           [  0.41929355]])
    >>> beta_hat, xptxpi = gls4tsls(reg.y, reg.z, reg.h, reg.u)
    >>> beta_hat
    array([[ 51.16882977],
           [ -1.12721019],
           [ -0.28543096],
           [  0.32904005]])
    >>> # run gls4tsls directly using 2SLS
    >>> reg=BaseGM_Lag(y, X, w, w_lags=2, robust='gls')
    >>> reg.betas
    array([[ 51.16882977],
           [ -1.12721019],
           [ -0.28543096],
           [  0.32904005]])

    """
    htz = np.dot(h.T, z)
    v = get_omega(h, u)
    vi = la.inv(v)
    htztvi = np.dot(htz.T, vi)
    htztvihtz = np.dot(htztvi, htz)
    htztvihtzi = la.inv(htztvihtz)    # [(h'Z)' omega^-1 (h'Z)]^-1
    hty = np.dot(h.T, y)
    htztvihty = np.dot(htztvi, hty)   #  (h'Z)' omega^-1 (h'y)
    betas = np.dot(htztvihtzi, htztvihty)
    return betas, htztvihtzi


def get_omega(rhs, u):
    u2 = u**2
    v = u2 * rhs
    return np.dot(rhs.T, v)           # weighting matrix (omega)

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

    Example with OLS and White
    
    >>> ols = BaseOLS(y,X, robust='white')
    >>> ols.vm
    array([[ 0.24491641,  0.01092258, -0.03438619],
           [ 0.01092258,  0.01796867, -0.00071345],
           [-0.03438619, -0.00071345,  0.00501042]])
    
    Example with OLS and HAC

    >>> wk = pysal.open("examples/kernel_knn15_epanechnikov_3085_random_points.gwt","r").read()
    >>> ols = BaseOLS(y,X, robust='hac')
    >>> ols.vm
    array([[ 0.17004545,  0.00226532, -0.02243898],
           [ 0.00226532,  0.00941319, -0.00031638],
           [-0.02243898, -0.00031638,  0.00313386]])

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



