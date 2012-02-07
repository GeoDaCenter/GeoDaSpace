import numpy as np
import numpy.linalg as la
from pysal.spreg.ols import RegressionProps
import robust as ROBUST
import pysal.spreg.user_output as USER
import gmm_utils as GMM

class GSTSLS_dev(RegressionProps):
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
                  array of external exogenous variables to use as instruments
                  (note: this should not contain any variables from x)
    w           : spatial weights object
                  pysal spatial weights object
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
                  array of independent variables (with constant added if
                  constant parameter set to True)
    y           : array
                  nx1 array of dependent variable
    z           : array
                  nxk array of variables (combination of x and yend)
    h           : array
                  nxl array of instruments (combination of x and q)
    yend        : array
                  endogenous variables (assumed to be aligned with y)
    q           : array
                  array of external exogenous variables
    lamb        : double
                  spatial autoregressive parameter
    betas       : array
                  kx1 array of estimated coefficients
    u           : array
                  nx1 array of residuals 
    predy       : array
                  nx1 array of predicted values 
    n           : integer
                  number of observations
    k           : int
                  Number of variables, including exogenous and endogenous variables
    kstar       : int
                  Number of endogenous variables. 
    zth         : array
                  z.T * h
    hth         : array
                  h.T * h
    htz         : array
                  h.T * z
    hthi        : array
                  inverse of h.T * h
    xp          : array
                  h * np.dot(hthi, htz)           
    xptxpi      : array
                  inverse of np.dot(xp.T,xp), used to compute vm
    pfora1a2    : array
                  used to compute a1, a2
    
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
    >>> w=pysal.open("examples/columbus.GAL").read() 
    >>> reg = GSTSLS_dev(X, y, yd, q, w, 0.1)
    >>> print reg.betas
    [[ 24.77541655]
     [ -1.68710261]
     [  0.24844278]]

    
    """
    def __init__(self, x, y, yend, q, w, lamb, constant=True, robust=None):
        
        self.y = GMM.get_spFilter(w, lamb, y)
        self.n = y.shape[0]
        self.x = GMM.get_spFilter(w, lamb, x)
        self.yend = GMM.get_spFilter(w, lamb,yend)
        self.kstar = yend.shape[1]
        
        self.z = np.hstack((self.x, self.yend))  # including exogenous and endogenous variables   
        self.h = np.hstack((x,q))   # including exogenous variables and instruments
        self.q = q
        
        if constant:
            self.z = np.hstack((np.ones(y.shape),self.z))
            self.h = np.hstack((np.ones(y.shape),self.h))

        self.k = self.z.shape[1]    # k = number of exogenous variables and endogenous variables 
        
        hth = np.dot(self.h.T,self.h)
        hthi = la.inv(hth)
        htz = np.dot(self.h.T,self.z)
        zth = np.dot(self.z.T,self.h)  
        
        
        factor_1 = np.dot(zth,hthi)
        factor_2 = np.dot(factor_1,self.h.T)
        factor_2 = np.dot(factor_2,self.z)
        factor_2 = la.inv(factor_2)        
        factor_2 = np.dot(factor_2,factor_1)        
        factor_2 = np.dot(factor_2,self.h.T)       
        betas = np.dot(factor_2,self.y)
        self.betas = betas
        
        # predicted values
        self.predy = np.dot(self.z,self.betas)
        
        # residuals
        u = self.y - self.predy
        self.u = u
        
        # attributes used in property 
        self.zth = zth
        self.hth = hth
        self.htz = htz
        self.hthi =hthi
        
        factor = np.dot(hthi, htz)
        xp = np.dot(self.h, factor)
        xptxp = np.dot(xp.T,xp)
        xptxpi = la.inv(xptxp)
        self.xp = xp
        self.xptxpi = xptxpi
        
        # pfora1a2
        factor_4 = np.dot(self.zth, factor)
        self.pfora1a2 = self.n*np.dot(factor, la.inv(factor_4))
        
        if robust == 'gls':
            self.betas, self.xptxpi = ROBUST.gls_dev(self.z, self.y, self.h, self.u)
            self.predy = np.dot(self.z, self.betas)   # using original data and GLS betas
            self.u = self.y - self.predy              # using original data and GLS betas
            ### need to verify the VM for the non-spatial case

        self._cache = {}
        RegressionProps()
        self.sig2 = self.sig2n        

def _test():
    import doctest
    doctest.testmod()

                     
if __name__ == '__main__':
    _test()    
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
    w=pysal.open("examples/columbus.GAL").read()  
    reg = GSTSLS_dev(X, y, yd, q, w, 0.1)
    print reg.z
    print reg.h 

       
