import numpy as np
import numpy.linalg as la
import ols as OLS
from ols import Regression_Props

class TSLS(Regression_Props):
    """
    2SLS class in one expression

    NOTE: no consistency checks
    
    Maximal Complexity: 

    Parameters
    ----------

    x           : array
                  array of independent variables, excluding endogenous
                  variables (assumed to be aligned with y)
    y           : array
                  nx1 array of dependent variable
    yend        : array
                  endogenous variables (assumed to be aligned with y)
    q           : array
                  array of instruments (assumed to be aligned with yend); 
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)

    robust      : string
                  If 'white' then a White consistent estimator of the
                  variance-covariance matrix is given. If 'gls' then
                  generalized least squares is performed resulting in new
                  coefficient estimates along with a new variance-covariance
                  matrix. 

    Attributes
    ----------

    x           : array
                  array of independent variables (assumed to be aligned with y)
    y           : array
                  nx1 array of dependent variable
    z           : array
                  n*k array of independent variables, including endogenous
                  variables (assumed to be aligned with y)
    h           : array
                  nxl array of instruments, typically this includes all
                  exogenous variables from x and instruments
    n           : integer
                  number of observations
    delta       : array
                  kx1 array with estimated coefficients
    u           : array
                  nx1 array of residuals 
    predy       : array
                  nx1 array of predicted values 
    k           : int
                  Number of variables, including exogenous and endorgenous variables
    xptxpi      : array
                  used to compute vm
    
    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("examples/columbus.dbf","r")
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
    >>> print reg.delta
    array([[ 88.46579584],
           [  0.5200379 ],
           [ -1.58216593]])
    
    """
    def __init__(self, y, x, yend, q, constant=True, robust=None):
        
        self.y = y  
        self.n = y.shape[0]
        self.x = x
        
        z = np.hstack((x,yend))  # including exogenous and endogenous variables      
        self.z = z
        self.k = z.shape[1]    # k = number of exogenous variables and endogenous variables 
        h = np.hstack((x,q))   # including exogenous variables and instruments
        self.h = h
        
        if constant:
            z = np.hstack((np.ones(y.shape),z))
            h = np.hstack((np.ones(y.shape),h))
                    
        zt = z.T
        ht = h.T
        hth = np.dot(ht,h)
        hthi = la.inv(hth)
        htz = np.dot(ht,z)
        zth = np.dot(zt,h)
        ztht = zth.T        
        
        factor_1 = np.dot(zth,hthi)
        factor_2 = np.dot(factor_1,ht)
        factor_2 = np.dot(factor_2,z)
        factor_2 = la.inv(factor_2)        
        factor_2 = np.dot(factor_2,factor_1)        
        factor_2 = np.dot(factor_2,ht)       
        delta = np.dot(factor_2,y)
        self.delta = delta
        
        # predicted values
        self.predy = np.dot(z,delta)
        
        # residuals
        u = y - self.predy
        self.u = u
        
        # attributes used in property 
        self.zth = zth
        self.hth = hth
        self.htz = htz
        self.hthi =hthi
        
        factor = np.dot(hthi, htz)
        xp = np.dot(h, factor)
        xpt = xp.T
        xptxp = np.dot(xpt,xp)
        xptxpi = la.inv(xptxp)
        self.xp = xp
        self.xpt = xpt
        self.xptxpi = xptxpi
        
        self._cache = {}
        OLS.Regression_Props()
        
    @property
    def m(self):
        if 'm' not in self._cache:
            xtxixt = np.dot(self.xptxpi,self.xpt)
            xxtxixt = np.dot(self.xp, xtxixt)
            self._cache['m'] = np.eye(self.n) - xxtxixt
        return self._cache['m']    
    
    @property
    def vm(self):
        if 'vm' not in self._cache:
            self._cache['vm'] = np.dot(self.sig2, self.xptxpi)
        return self._cache['vm']
        
    @property
    def pfora1a2(self):
        if 'pfora1a2' not in self._cache:
            factor1 = self.zth/(n * 1.0)
            factor2 = n * self.hthi
            factor3 = self.htz/(n * 1.0)
            factor4 = np.dot(factor1, factor2)
            factor5 = np.dot(factor4, factor3)
            self._cache['pfora1a2'] = la.inv(factor5)
        return self._cache['pfora1a2']
        
        
if __name__ == '__main__':
    import numpy as np
    import pysal
    db=pysal.open("examples/columbus.dbf","r")
    y = np.array(db.by_col("CRIME"))
    y = np.reshape(y, (49,1))
    X = []
    X.append(db.by_col("INC"))
    X = np.array(X).T
    yd = []
    yd.append(db.by_col("HOVAL"))
    yd = np.array(yd).T
    # instrument for HOVAL with DISCBD
    q = []
    q.append(db.by_col("DISCBD"))
    q = np.array(q).T
    reg = TSLS(y, X, yd, q)
    print reg.delta
    print reg.vm 

       