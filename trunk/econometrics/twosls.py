import numpy as np
import numpy.linalg as la
from ols import RegressionProps
import robust as ROBUST
import user_output as USER

class BaseTSLS(RegressionProps):
    """
    2SLS class in one expression

    Parameters
    ----------

    y           : array
                  nx1 array of dependent variable
    x           : array
                  array of independent variables, excluding endogenous
                  variables
    yend        : array
                  endogenous variables
    q           : array
                  array of external exogenous variables to use as instruments
                  (note: this should not contain any variables from x)
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

    y           : array
                  nx1 array of dependent variable
    x           : array
                  array of independent variables (with constant added if
                  constant parameter set to True)
    z           : array
                  nxk array of variables (combination of x and yend)
    h           : array
                  nxl array of instruments (combination of x and q)
    yend        : array
                  endogenous variables
    q           : array
                  array of external exogenous variables
    betas       : array
                  kx1 array of estimated coefficients
    u           : array
                  nx1 array of residuals 
    predy       : array
                  nx1 array of predicted values 
    n           : integer
                  number of observations
    k           : int
                  Number of variables, including exogenous and endogenous
                  variables and constant
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
    >>> reg = BaseTSLS(y, X, yd, q)
    >>> print reg.betas
    [[ 88.46579584]
     [  0.5200379 ]
     [ -1.58216593]]
    
    """
    def __init__(self, y, x, yend, q, constant=True, robust=None):
        
        self.y = y  
        self.n = y.shape[0]
        self.x = x
        self.kstar = yend.shape[1]
        
        z = np.hstack((x,yend))  # including exogenous and endogenous variables   
        h = np.hstack((x,q))   # including exogenous variables and instruments
        
        if constant:
            z = np.hstack((np.ones(y.shape),z))
            h = np.hstack((np.ones(y.shape),h))

        self.z = z
        self.h = h
        self.q = q
        self.yend = yend
        self.k = z.shape[1]    # k = number of exogenous variables and endogenous variables 
        
        hth = np.dot(h.T,h)
        hthi = la.inv(hth)
        htz = np.dot(h.T,z)
        zth = np.dot(z.T,h)  
        
        
        factor_1 = np.dot(zth,hthi)
        factor_2 = np.dot(factor_1,h.T)
        factor_2 = np.dot(factor_2,z)
        factor_2 = la.inv(factor_2)        
        factor_2 = np.dot(factor_2,factor_1)        
        factor_2 = np.dot(factor_2,h.T)       
        betas = np.dot(factor_2,y)
        self.betas = betas
        
        # predicted values
        self.predy = np.dot(z,betas)
        
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
        xptxp = np.dot(xp.T,xp)
        xptxpi = la.inv(xptxp)
        self.xp = xp
        self.xptxpi = xptxpi
        
        # pfora1a2
        factor_4 = np.dot(self.zth, factor)
        self.pfora1a2 = self.n*np.dot(factor, la.inv(factor_4))
        
        if robust == 'gls':
            self.betas, self.xptxpi = ROBUST.gls4tsls(y, z, h, self.u)
            self.predy = np.dot(z, self.betas)   # using original data and GLS betas
            self.u = y - self.predy              # using original data and GLS betas
            ### need to verify the VM for the non-spatial case

        self._cache = {}
        RegressionProps()
        self.sig2 = self.sig2n
        
    @property
    def m(self):
        if 'm' not in self._cache:
            xtxixt = np.dot(self.xptxpi,self.xp.T)
            xxtxixt = np.dot(self.xp, xtxixt)
            self._cache['m'] = np.eye(self.n) - xxtxixt
        return self._cache['m']    
    
    @property
    def vm(self):
        if 'vm' not in self._cache:
            self._cache['vm'] = np.dot(self.sig2, self.xptxpi)
        return self._cache['vm']


class TSLS(BaseTSLS, USER.DiagnosticBuilder):
    """
    need test requiring BOTH yend and q

    Parameters
    ----------

    y           : array
                  nx1 array of dependent variable
    x           : array
                  array of independent variables, excluding endogenous
                  variables
    w           : spatial weights object
                  if provided then spatial diagnostics are computed       
    yend        : array
                  endogenous variables
    q           : array
                  array of external exogenous variables to use as instruments
                  (note: this should not contain any variables from x)
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)            
    robust      : string
                  If 'white' then a White consistent estimator of the
                  variance-covariance matrix is given. If 'gls' then
                  generalized least squares is performed resulting in new
                  coefficient estimates along with a new variance-covariance
                  matrix. 
    name_y      : string
                  Name of dependent variables for use in output
    name_x      : list of strings
                  Names of independent variables for use in output
    name_yend   : list of strings
                  Names of endogenous variables for use in output
    name_q      : list of strings
                  Names of instruments for use in output
    name_ds     : string
                  Name of dataset for use in output
    vm          : boolean
                  If True, include variance matrix in summary results
    pred        : boolean
                  If True, include y, predicted values and residuals in summary results

    Attributes
    ----------

    y           : array
                  nx1 array of dependent variable
    x           : array
                  array of independent variables (with constant added if
                  constant parameter set to True)
    z           : array
                  nxk array of variables (combination of x and yend)
    h           : array
                  nxl array of instruments (combination of x and q)
    yend        : array
                  endogenous variables
    q           : array
                  array of external exogenous variables
    betas       : array
                  kx1 array of estimated coefficients
    u           : array
                  nx1 array of residuals 
    predy       : array
                  nx1 array of predicted values 
    n           : integer
                  number of observations
    k           : int
                  Number of variables, including exogenous and endogenous
                  variables and constant
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
    >>> reg = TSLS(y, X, yd, q, name_x=['inc'], name_y='crime', name_yend=['hoval'], name_q=['discbd'], name_ds='columbus')
    >>> print reg.betas
    [[ 88.46579584]
     [  0.5200379 ]
     [ -1.58216593]]
    

    """
    def __init__(self, y, x, yend, q, w=None, constant=True, name_y=None,\
                        name_x=None, name_yend=None, name_q=None,\
                        name_ds=None, robust=None, vm=False,\
                        pred=False):
        BaseTSLS.__init__(self, y, x, yend, q, constant, robust)
        self.title = "TWO STAGE LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_q = USER.set_name_q(name_q, q)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        USER.DiagnosticBuilder.__init__(self, x=x, constant=constant, w=w,\
                                            vm=vm, pred=pred, instruments=True)
        
        ### tsls.summary output needs to be checked. Currently uses z-stats
        ### and prints names of instruments. Need to replace R squared with
        ### some pseudo version. Need to add multicollinearity test back in.
        ### Need to bring in correct f_stat.






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
    reg = BaseTSLS(y, X, yd, q)
    print reg.betas
    print reg.vm 

       
