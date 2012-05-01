import numpy as np
import copy
import numpy.linalg as la
import robust as ROBUST
import user_output as USER
from utils import spdot, sphstack, RegressionPropsY, RegressionPropsVM

__author__ = "Luc Anselin luc.anselin@asu.edu, David C. Folch david.folch@asu.edu, Jing Yao jingyao@asu.edu"
__all__ = ["TSLS"]

class BaseTSLS(RegressionPropsY, RegressionPropsVM):
    """
    Two stage least squares (2SLS) (note: no consistency checks or
    diagnostics)

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (note: 
                   this should not contain any variables from x); cannot be
                   used in combination with h
    h            : array
                   Two dimensional array with n rows and one column for each
                   exogenous variable to use as instruments (note: this 
                   can contain variables from x); cannot be used in 
                   combination with q
    robust       : string
                   If 'white', then a White consistent estimator of the
                   variance-covariance matrix is given.  If 'hac', then a
                   HAC consistent estimator of the variance-covariance
                   matrix is given. Default set to None. 
    gwk          : pysal W object
                   Kernel spatial weights needed for HAC estimation. Note:
                   matrix must have ones along the main diagonal.
    sig2n_k      : boolean
                   If True, then use n-k to estimate sigma^2. If False, use n.


    Attributes
    ----------
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    kstar        : integer
                   Number of endogenous variables. 
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments 
    z            : array
                   nxk array of variables (combination of x and yend)
    h            : array
                   nxl array of instruments (combination of x and q)
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
    sig2n        : float
                   Sigma squared (computed with n in the denominator)
    sig2n_k      : float
                   Sigma squared (computed with n-k in the denominator)
    hth          : float
                   H'H
    hthi         : float
                   (H'H)^-1
    varb         : array
                   (Z'H (H'H)^-1 H'Z)^-1
    zthhthi      : array
                   Z'H(H'H)^-1
    pfora1a2     : array
                   n(zthhthi)'varb

    
    Examples
    --------

    >>> import numpy as np
    >>> import pysal
    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')
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
    def __init__(self, y, x, yend, q=None, h=None, constant=True,\
                 robust=None, gwk=None, sig2n_k=False):

        if issubclass(type(q), np.ndarray) and issubclass(type(h), np.ndarray):  
            raise Exception, "Please do not provide 'q' and 'h' together"
        if q==None and h==None:
            raise Exception, "Please provide either 'q' or 'h'"
        
        self.y = y  
        self.n = y.shape[0]

        if constant:
            self.x = sphstack(np.ones(y.shape),x)
        else:
            self.x = x

        self.kstar = yend.shape[1]        
        z = sphstack(self.x,yend)  # including exogenous and endogenous variables   
        if type(h).__name__ != 'ndarray':
            h = sphstack(self.x,q)   # including exogenous variables and instrument

        self.z = z
        self.h = h
        self.q = q
        self.yend = yend
        self.k = z.shape[1]    # k = number of exogenous variables and endogenous variables 
        
        hth = spdot(h.T,h)    
        hthi = la.inv(hth)
        zth = spdot(z.T,h)    
        hty = spdot(h.T,y) 
        
        factor_1 = np.dot(zth,hthi)  
        factor_2 = np.dot(factor_1,zth.T)  
        varb = la.inv(factor_2)          # this one needs to be in cache to be used in AK
        factor_3 = np.dot(varb,factor_1)   
        #print factor_3.shape, hty.shape
        betas = np.dot(factor_3,hty)  
        self.betas = betas
        self.varb = varb
        self.zthhthi = factor_1  
        
        # predicted values
        self.predy = spdot(z,betas)
        
        # residuals
        u = y - self.predy
        self.u = u
        
        # attributes used in property 
        self.hth = hth     # Required for condition index
        self.hthi =hthi    # Used in error models
      
        if robust:
            self.vm = ROBUST.robust_vm(reg=self, gwk=gwk)

        self._cache = {}
        if sig2n_k:
            self.sig2 = self.sig2n_k
        else:
            self.sig2 = self.sig2n

    @property
    def pfora1a2(self):
        if 'pfora1a2' not in self._cache:
            self._cache['pfora1a2'] = self.n*np.dot(self.zthhthi.T, self.varb) 
        return self._cache['pfora1a2']    
            
    @property
    def vm(self):
        if 'vm' not in self._cache:
            self._cache['vm'] = np.dot(self.sig2, self.varb)
        return self._cache['vm']

class TSLS(BaseTSLS, USER.DiagnosticBuilder):
    """
    Two stage least squares with results and diagnostics.

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (note: 
                   this should not contain any variables from x)
    w            : pysal W object
                   Spatial weights object (required if running spatial
                   diagnostics)
    robust       : string
                   If 'white', then a White consistent estimator of the
                   variance-covariance matrix is given.  If 'hac', then a
                   HAC consistent estimator of the variance-covariance
                   matrix is given. Default set to None. 
    gwk          : pysal W object
                   Kernel spatial weights needed for HAC estimation. Note:
                   matrix must have ones along the main diagonal.
    sig2n_k      : boolean
                   If True, then use n-k to estimate sigma^2. If False, use n.
    spat_diag    : boolean
                   If True, then compute Anselin-Kelejian test (requires w)
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_q       : list of strings
                   Names of instruments for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_gwk     : string
                   Name of kernel weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output


    Attributes
    ----------
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    kstar        : integer
                   Number of endogenous variables. 
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments 
    z            : array
                   nxk array of variables (combination of x and yend)
    h            : array
                   nxl array of instruments (combination of x and q)
    robust       : string
                   Adjustment for robust standard errors
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
    std_err      : array
                   1xk array of standard errors of the betas    
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    ak_test      : tuple
                   Anselin-Kelejian test; tuple contains the pair (statistic,
                   p-value)
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_z       : list of strings
                   Names of exogenous and endogenous variables for use in 
                   output
    name_q       : list of strings
                   Names of external instruments
    name_h       : list of strings
                   Names of all instruments used in ouput
    name_w       : string
                   Name of weights matrix for use in output
    name_gwk     : string
                   Name of kernel weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    title        : string
                   Name of the regression method used
    sig2n        : float
                   Sigma squared (computed with n in the denominator)
    sig2n_k      : float
                   Sigma squared (computed with n-k in the denominator)
    hth          : float
                   H'H
    hthi         : float
                   (H'H)^-1
    varb         : array
                   (Z'H (H'H)^-1 H'Z)^-1
    zthhthi      : array
                   Z'H(H'H)^-1
    pfora1a2     : array
                   n(zthhthi)'varb

    
    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import numpy as np
    >>> import pysal

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> db = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')
    
    Extract the CRIME column (crime rates) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Extract INC (income) vector from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this model adds a vector of ones to the
    independent variables passed in, but this can be overridden by passing
    constant=False.

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T

    In this case we consider HOVAL (home value) is an endogenous regressor.
    We tell the model that this is so by passing it in a different parameter
    from the exogenous variables (x).

    >>> yd = []
    >>> yd.append(db.by_col("HOVAL"))
    >>> yd = np.array(yd).T

    Because we have endogenous variables, to obtain a correct estimate of the
    model, we need to instrument for HOVAL. We use DISCBD (distance to the
    CBD) for this and hence put it in the instruments parameter, 'q'.

    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T

    We are all set with the preliminars, we are good to run the model. In this
    case, we will need the variables (exogenous and endogenous) and the
    instruments. If we want to have the names of the variables printed in the
    output summary, we will have to pass them in as well, although this is optional.

    >>> reg = TSLS(y, X, yd, q, name_x=['inc'], name_y='crime', name_yend=['hoval'], name_q=['discbd'], name_ds='columbus')
    >>> print reg.betas
    [[ 88.46579584]
     [  0.5200379 ]
     [ -1.58216593]]

    """
    def __init__(self, y, x, yend, q,\
                 w=None,\
                 robust=None, gwk=None, sig2n_k=False,\
                 spat_diag=False,\
                 vm=False, name_y=None, name_x=None,\
                 name_yend=None, name_q=None,\
                 name_w=None, name_gwk=None, name_ds=None):

        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        USER.check_robust(robust, gwk)
        USER.check_spat_diag(spat_diag, w)
        USER.check_constant(x)
        BaseTSLS.__init__(self, y=y, x=x, yend=yend, q=q,\
                              robust=robust, gwk=gwk, sig2n_k=sig2n_k)
        self.title = "TWO STAGE LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_q = USER.set_name_q(name_q, q)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.robust = USER.set_robust(robust)
        self.name_w = USER.set_name_w(name_w, w)
        self.name_gwk = USER.set_name_w(name_gwk, gwk)
        self._get_diagnostics(w=w, beta_diag=True, nonspat_diag=False,\
                                    spat_diag=spat_diag, vm=vm,
                                    std_err=self.robust)

    def _get_diagnostics(self, beta_diag=True, w=None, nonspat_diag=True,\
                              spat_diag=False, vm=False, moran=False,
                              std_err=None):
        USER.DiagnosticBuilder.__init__(self, w=w, beta_diag=beta_diag,\
                                            nonspat_diag=nonspat_diag,\
                                            spat_diag=spat_diag, vm=vm,\
                                            instruments=True,
                                            moran=False, std_err=std_err)
        

                     
if __name__ == '__main__':
    import doctest
    doctest.testmod()
