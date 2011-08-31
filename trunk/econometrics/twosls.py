import numpy as np
import copy
import numpy.linalg as la
import robust as ROBUST
import user_output as USER
from utils import RegressionProps

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
                  array of external exogenous variables to use as instruments;
                  (note: this should not contain any variables from x; all x
                  variables will be added by default as instruments (see h))
                  (note: user must provide either q or h, but not both)
    h           : array
                  array of all exogenous variables to use as instruments
                  (note: this is the entire instrument set, no additional
                  variables will be added (see q))
                  (note: user must provide either q or h, but not both)
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)
    sig2n_k     : boolean
                  Whether to use n-k (if True) or n (if False, default) to
                  estimate sigma2                  
    robust      : string
                  If 'white' or 'hac' then a White consistent or HAC estimator of the
                  variance-covariance matrix is given. If 'gls' then
                  generalized least squares is performed resulting in new
                  coefficient estimates along with a new variance-covariance
                  matrix. 
    wk          : spatial weights object
                  pysal kernel weights object

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
    
    Notes
    -----
    Sigma squared calculated using n in the denominator. 
    
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
    >>> reg = BaseTSLS(y, X, yd, q=q)
    >>> print reg.betas
    [[ 88.46579584]
     [  0.5200379 ]
     [ -1.58216593]]
    >>> reg = BaseTSLS(y, X, yd, q=q, robust="white")
    
    """
    def __init__(self, y, x, yend, q=None, h=None, constant=True, sig2n_k=False,\
                        robust=None, wk=None):
        
        if issubclass(type(q), np.ndarray) and issubclass(type(h), np.ndarray):  
            raise Exception, "Please do not provide 'q' and 'h' together"
        if q==None and h==None:
            raise Exception, "Please provide either 'q' or 'h'"
        
        self.y = y  
        self.n = y.shape[0]

        if constant:
            self.x = np.hstack((np.ones(y.shape),x))
        else:
            self.x = x

        self.kstar = yend.shape[1]        
        z = np.hstack((self.x,yend))  # including exogenous and endogenous variables   
        if type(h).__name__ != 'ndarray':
            h = np.hstack((self.x,q))   # including exogenous variables and instrument

        self.z = z
        self.h = h
        self.q = q
        self.yend = yend
        self.k = z.shape[1]    # k = number of exogenous variables and endogenous variables 
        
        hth = np.dot(h.T,h)    #H'H
        hthi = la.inv(hth)
#        htz = np.dot(h.T,z)    #H'Z  not needed
        zth = np.dot(z.T,h)    
        hty = np.dot(h.T,y)     #H'y
        
        factor_1 = np.dot(zth,hthi)  #LA Z'H (H'H)^-1   np.dot(htz.T,hthi)
#        factor_2 = np.dot(factor_1,htz)   # (Z'H (H'H)^-1 H'Z)
        factor_2 = np.dot(factor_1,zth.T)  # LA added
        varb = la.inv(factor_2)          # this one needs to be in cache to be used in AK
#        factor_2 = np.dot(varb,factor_1)  # LA (Z'H (H'H)^-1 H'Z)^-1 Z'H (H'H)^-1  -- do not reuse factor_2, define new
        factor_3 = np.dot(varb,factor_1)   #LA
#        betas = np.dot(factor_2,hty)         # (Z'H (H'H)^-1 H'Z)^-1 Z'H (H'H)^-1 H'y -- LA removed
        betas = np.dot(factor_3,hty)  #LA
        self.betas = betas
        self.varb = varb
        self.zthhthi = factor_1   # need this for HAC
        
        # predicted values
        self.predy = np.dot(z,betas)
        
        # residuals
        u = y - self.predy
        self.u = u
        
        # attributes used in property 
        self.zth = zth
        self.hthi =hthi
        
        xp = np.dot(h, self.zthhthi.T)    # H (H'H)^-1 H'Z 
#        xptxp = np.dot(xp.T,xp)    # LA Z'H (H'H)^-1 H' H (H'H)^-1 H'Z  -- note that the H'H cancel out, not needed
#        xptxpi = la.inv(xptxp)        # LA is really (Z'H (H'H)^-1 H'Z)^1 -- already there as varb!
        self.xp = xp
#        self.xptxpi = xptxpi  # LA remove
        self.xptxpi = varb
      
        if robust:
            self.vm = ROBUST.robust_vm(reg=self, wk=wk)

        self._cache = {}
        RegressionProps()
        if sig2n_k:
            self.sig2 = self.sig2n_k
        else:
            self.sig2 = self.sig2n

    @property
    def pfora1a2(self):
        if 'pfora1a2' not in self._cache:
            # pfora1a2
#            factor_4 = np.dot(self.zth, self.zthhthi.T)         # LA  Z'H (H'H)^-1 H'Z  #note is factor 2 above
#          self._cache['pfora1a2'] = self.n*np.dot(self.zthhthi.T, la.inv(factor_4))  # LA inverse is already varb
            self._cache['pfora1a2'] = self.n*np.dot(self.zthhthi.T,self.varb)  #LA
        return self._cache['pfora1a2']    
            
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
    sig2n_k     : boolean
                  Whether to use n-k (if True) or n (if False, default) to
                  estimate sigma2                  
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

    Notes
    -----
    Sigma squared calculated using n in the denominator. 
    
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
    def __init__(self, y, x, yend, q, w=None, constant=True, sig2n_k=False,\
                        robust=None, wk=None,\
                        nonspat_diag=True, spat_diag=False,\
                        name_y=None, name_x=None, name_yend=None, name_q=None,\
                        name_ds=None, vm=False, pred=False):
        #### we currently ignore nonspat_diag parameter ####

        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        USER.check_robust(robust, wk)
        USER.check_spat_diag(spat_diag, w)
        BaseTSLS.__init__(self, y=y, x=x, yend=yend, q=q, constant=constant,\
                              robust=robust, wk=wk, sig2n_k=sig2n_k)
        self.title = "TWO STAGE LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_q = USER.set_name_q(name_q, q)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.robust = USER.set_robust(robust)
        #### we currently ignore nonspat_diag parameter ####
        self._get_diagnostics(w=w, beta_diag=True, nonspat_diag=False,\
                                    spat_diag=spat_diag, vm=vm, pred=pred)

    def _get_diagnostics(self, beta_diag=True, w=None, nonspat_diag=True,\
                              spat_diag=False, vm=False, pred=False):
        USER.DiagnosticBuilder.__init__(self, w=w, beta_diag=True,\
                                            nonspat_diag=nonspat_diag,\
                                            spat_diag=spat_diag, vm=vm,\
                                            pred=pred, instruments=True)
        

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
    reg = BaseTSLS(y, X, yd, q=q, robust='white')
    print reg.betas
    print reg.vm 

       
