import copy
import numpy as np
import pysal
import numpy.linalg as la
import twosls as TSLS
import robust as ROBUST
import user_output as USER
from utils import get_lags, set_endog, sp_att

class BaseGM_Lag(TSLS.BaseTSLS):
    """
    Spatial two stage least squares (S2SLS) (note: no consistency checks or
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
    w            : pysal W object
                   Spatial weights object (note: if provided then spatial
                   diagnostics are computed)   
    w_lags       : integer
                   Orders of W to include as instruments for the spatially
                   lagged dependent variable. For example, w_lags=1, then
                   instruments are WX; if w_lags=2, then WX, WWX; and so on.
    lag_q        : boolean
                   If True, then include spatial lags of the additional 
                   instruments (q).
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
    >>> import pysal.spreg.diagnostics as D
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> # no non-spatial endogenous variables
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T
    >>> reg=BaseGM_Lag(y, X, w=w, w_lags=2)
    >>> reg.betas
    array([[  4.53017056e+01],
           [  6.20888617e-01],
           [ -4.80723451e-01],
           [  2.83622122e-02]])
    >>> D.se_betas(reg)
    array([ 17.91278862,   0.52486082,   0.1822815 ,   0.31740089])
    >>> reg=BaseGM_Lag(y, X, w=w, w_lags=2, robust='white')
    >>> reg.betas
    array([[  4.53017056e+01],
           [  6.20888617e-01],
           [ -4.80723451e-01],
           [  2.83622122e-02]])
    >>> D.se_betas(reg)
    array([ 20.47077481,   0.50613931,   0.20138425,   0.38028295])
    >>> # instrument for HOVAL with DISCBD
    >>> X = np.array(db.by_col("INC"))
    >>> X = np.reshape(X, (49,1))
    >>> yd = np.array(db.by_col("CRIME"))
    >>> yd = np.reshape(yd, (49,1))
    >>> q = np.array(db.by_col("DISCBD"))
    >>> q = np.reshape(q, (49,1))
    >>> reg=BaseGM_Lag(y, X, w=w, yend=yd, q=q, w_lags=2)

    References
    ----------

    .. [1] Kelejian, H.H., Prucha, I.R. and Yuzefovich, Y. (2004)
    "Instrumental variable estimation of a spatial autoregressive model with
    autoregressive disturbances: large and small sample results". Advances in
    Econometrics, 18, 163-198.
    """

    def __init__(self, y, x, yend=None, q=None,\
                 w=None, w_lags=1, lag_q=True,\
                 robust=None, gwk=None, sig2n_k=False):

        yend2, q2 = set_endog(y, x, w, yend, q, w_lags, lag_q)
        TSLS.BaseTSLS.__init__(self, y, x, yend2, q=q2,\
                               sig2n_k=sig2n_k)        
        if robust:
            self.vm = ROBUST.robust_vm(self, gwk=gwk)


class GM_Lag(BaseGM_Lag, USER.DiagnosticBuilder):
    """
    Spatial two stage least squares (S2SLS) with results and diagnostics.

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
    w            : pysal W object
                   Spatial weights object 
    w_lags       : integer
                   Orders of W to include as instruments for the spatially
                   lagged dependent variable. For example, w_lags=1, then
                   instruments are WX; if w_lags=2, then WX, WWX; and so on.
    lag_q        : boolean
                   If True, then include spatial lags of the additional 
                   instruments (q).
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
                   If True, then compute Anselin-Kelejian test
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
    e            : array
                   nx1 array of residuals (using reduced form)
    predy        : array
                   nx1 array of predicted y values
    predy_e      : array
                   nx1 array of predicted y values (using reduced form)
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
    pr2_e        : float
                   Pseudo R squared (squared correlation between y and ypred_e
                   (using reduced form))
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

    >>> import numpy as np
    >>> import pysal
    >>> import pysal.spreg.diagnostics as D
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> # no non-spatial endogenous variables
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T
    >>> reg=GM_Lag(y, X, w=w, w_lags=2, name_x=['inc', 'crime'], name_y='hoval', name_ds='columbus')
    >>> reg.betas
    array([[  4.53017056e+01],
           [  6.20888617e-01],
           [ -4.80723451e-01],
           [  2.83622122e-02]])
    >>> D.se_betas(reg)
    array([ 17.91278862,   0.52486082,   0.1822815 ,   0.31740089])
    >>> reg=GM_Lag(y, X, w=w, w_lags=2, robust='white', name_x=['inc', 'crime'], name_y='hoval', name_ds='columbus')
    >>> reg.betas
    array([[  4.53017056e+01],
           [  6.20888617e-01],
           [ -4.80723451e-01],
           [  2.83622122e-02]])
    >>> reg.std_err
    array([ 20.47077481,   0.50613931,   0.20138425,   0.38028295])
    >>> # instrument for HOVAL with DISCBD
    >>> X = np.array(db.by_col("INC"))
    >>> X = np.reshape(X, (49,1))
    >>> yd = np.array(db.by_col("CRIME"))
    >>> yd = np.reshape(yd, (49,1))
    >>> q = np.array(db.by_col("DISCBD"))
    >>> q = np.reshape(q, (49,1))
    >>> reg=GM_Lag(y, X, w=w, yend=yd, q=q, w_lags=2, name_x=['inc'], name_y='hoval', name_yend=['crime'], name_q=['discbd'], name_ds='columbus')


    """
    def __init__(self, y, x, yend=None, q=None,\
                 w=None, w_lags=1, lag_q=True,\
                 robust=None, gwk=None, sig2n_k=False,\
                 spat_diag=False,\
                 vm=False, name_y=None, name_x=None,\
                 name_yend=None, name_q=None,\
                 name_w=None, name_gwk=None, name_ds=None):

        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        USER.check_robust(robust, gwk)
        USER.check_constant(x)
        BaseGM_Lag.__init__(self, y=y, x=x, w=w, yend=yend, q=q,\
                            w_lags=w_lags, robust=robust,\
                            lag_q=lag_q, sig2n_k=sig2n_k)
        self.predy_e, self.e = sp_att(w,self.y,self.predy,\
                      self.z[:,-1].reshape(self.n,1),self.betas[-1])
        self.title = "SPATIAL TWO STAGE LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_yend.append(USER.set_name_yend_sp(self.name_y))
        self.name_z = self.name_x + self.name_yend
        self.name_q = USER.set_name_q(name_q, q)
        self.name_q.extend(USER.set_name_q_sp(self.name_x, w_lags, self.name_q, lag_q))
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.robust = USER.set_robust(robust)
        self.name_w = USER.set_name_w(name_w, w)
        self.name_gwk = USER.set_name_w(name_gwk, gwk)
        self._get_diagnostics(w=w, beta_diag=True, nonspat_diag=False,\
                                    vm=vm, spat_diag=spat_diag,
                                    std_err=self.robust)

    def _get_diagnostics(self, beta_diag=True, w=None, nonspat_diag=True,\
                              spat_diag=False, vm=False, std_err=None):
        USER.DiagnosticBuilder.__init__(self, w=w, beta_diag=beta_diag,\
                                            nonspat_diag=nonspat_diag,\
                                            spat_diag=spat_diag, vm=vm,\
                                            instruments=True, std_err=std_err,\
                                            spatial_lag=True)

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
    import numpy as np
    import pysal
    import pysal.spreg.diagnostics as D
    w = pysal.rook_from_shapefile("examples/columbus.shp")
    w.transform = 'r'
    db=pysal.open("examples/columbus.dbf","r")
    y = np.array(db.by_col("HOVAL"))
    y = np.reshape(y, (49,1))
    # no non-spatial endogenous variables
    X = []
    X.append(db.by_col("INC"))
    X.append(db.by_col("CRIME"))
    X = np.array(X).T
    reg=BaseGM_Lag(y, X, w=w, w_lags=2)
    print reg.betas
    print reg.vm
    reg=BaseGM_Lag(y, X, w=w, w_lags=2, robust='white')
    print reg.betas
    print reg.vm
    from robust import robust_vm
    print robust_vm(reg)
