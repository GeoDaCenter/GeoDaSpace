"""
Spatial Error Models module
"""
from scipy.stats import norm
from scipy import sparse as SP
import numpy as np
import multiprocessing as mp
import copy
from numpy import linalg as la
import pysal.spreg.ols as OLS
from pysal.spreg.diagnostics import se_betas
from pysal import lag_spatial
from utils import power_expansion
from utils import get_A1_hom, get_A2_hom, get_A1_het, optim_moments, get_spFilter, get_lags, _moments2eqs
from utils import RegressionProps
import twosls as TSLS
import user_output as USER



class BaseGM_Error(RegressionProps):
    """
    Generalized Moments Spatially Weighted Least Squares (OLS + GMM) as in Kelejian and Prucha
    (1998) [1]_ and Kelejian and Prucha (1999) [2]_
    ...

    Parameters
    ----------

    y           : array
                  nx1 array of dependent variable
    x           : array
                  nxk array of independent variables (assumed to be aligned with y)
    w           : W
                  Spatial weights instance 
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)

    Attributes
    ----------

    betas       : array
                  kx1 array with estimated coefficients (including spatial
                  parameter)
    se_betas    : array
                  kx1 array with standard errors for estimated coefficients
                  NOTE: it corrects by sqrt( (n-k)/n ) as in R's spdep
    z           : array
                  kx1 array with estimated coefficients divided by the standard errors
    pvals       : array
                  kx1 array with p-values of the estimated coefficients
    u           : array
                  Vector of residuals
    sig2        : float
                  Sigma squared for the residuals of the transformed model (as
                  in R's spdep)
    step2OLS    : ols
                  Regression object from the OLS step with spatially filtered
                  variables

    References
    ----------

    .. [1] Kelejian, H.R., Prucha, I.R. (1998) "A generalized spatial
    two-stage least squares procedure for estimating a spatial autoregressive
    model with autoregressive disturbances". The Journal of Real State
    Finance and Economics, 17, 1.

    .. [2] Kelejian, H.R., Prucha, I.R. (1999) "A Generalized Moments
    Estimator for the Autoregressive Parameter in a Spatial Model".
    International Economic Review, 40, 2.

    Examples
    --------

    >>> import pysal
    >>> import numpy as np
    >>> dbf = pysal.open('examples/columbus.dbf','r')
    >>> y = np.array([dbf.by_col('HOVAL')]).T
    >>> x = np.array([dbf.by_col('INC'), dbf.by_col('CRIME')]).T
    >>> w = pysal.open('examples/columbus.gal', 'r').read() 
    >>> w.transform='r'
    >>> model = BaseGM_Error(y, x, w)
    >>> np.around(model.betas, decimals=6)
    array([[ 47.694634],
           [  0.710453],
           [ -0.550527],
           [  0.32573 ]])
    >>> np.around(model.se_betas, decimals=6)
    array([[ 12.412039],
           [  0.504443],
           [  0.178496]])
    >>> np.around(model.z, decimals=6)
    array([[ 3.842611],
           [ 1.408391],
           [-3.084247]])
    >>> np.around(model.pvals, decimals=6)
    array([[  1.22000000e-04],
           [  1.59015000e-01],
           [  2.04100000e-03]])
    >>> np.around(model.sig2, decimals=6)
    198.559595

    """
    def __init__(self, y, x, w, constant=True):

        #1a. OLS --> \tilde{betas}
        ols = OLS.BaseOLS(y, x, constant=constant)
        self.n, self.k = ols.x.shape
        self.x = ols.x
        self.y = ols.y

        #1b. GMM --> \tilde{\lambda1}
        moments = _momentsGM_Error(w, ols.u)
        lambda1 = optim_moments(moments)

        #2a. OLS -->\hat{betas}
        xs = get_spFilter(w, lambda1, self.x)
        ys = get_spFilter(w, lambda1, self.y)
        ols2 = OLS.BaseOLS(ys, xs, constant=False)

        #Output
        self.u = y - np.dot(self.x, ols2.betas)
        self.betas = np.vstack((ols2.betas, np.array([[lambda1]])))
        self.sig2 = ols2.sig2n

        self.vm = self.sig2 * ols2.xtxi
        se_betas = np.sqrt(self.vm.diagonal())
        self.se_betas = se_betas.reshape((len(ols2.betas), 1))
        zs = ols2.betas / self.se_betas
        pvals = norm.sf(abs(zs)) * 2.
        self.z, self.pvals = zs, pvals
        
        self.step2OLS = ols2
        self._cache = {}

class GM_Error(BaseGM_Error, USER.DiagnosticBuilder):
    """


    Examples
    --------

    >>> import pysal
    >>> import numpy as np
    >>> dbf = pysal.open('examples/columbus.dbf','r')
    >>> y = np.array([dbf.by_col('HOVAL')]).T
    >>> x = np.array([dbf.by_col('INC'), dbf.by_col('CRIME')]).T
    >>> w = pysal.open('examples/columbus.gal', 'r').read() 
    >>> w.transform='r'
    >>> #model = GM_Error(y, x, w, name_y='hoval', name_x=['income', 'crime'], name_ds='columbus')
    >>> #print model.name_x
    ['CONSTANT', 'income', 'crime', 'lambda']
    >>> #np.around(model.betas, decimals=6)
    array([[ 47.694634],
           [  0.710453],
           [ -0.550527],
           [  0.32573 ]])
    >>> #np.around(model.se_betas, decimals=6)
    array([[ 12.412039],
           [  0.504443],
           [  0.178496]])
    >>> #np.around(model.z, decimals=6)
    array([[ 3.842611],
           [ 1.408391],
           [-3.084247]])
    >>> #np.around(model.pvals, decimals=6)
    array([[  1.22000000e-04],
           [  1.59015000e-01],
           [  2.04100000e-03]])
    >>> #np.around(model.sig2, decimals=6)
    198.559595

    """
    def __init__(self, y, x, w, constant=True, nonspat_diag=True,\
                        name_y=None, name_x=None, name_ds=None,\
                        vm=False, pred=False):                
        #### we currently ignore nonspat_diag parameter ####

        USER.check_arrays(y, x)
        USER.check_weights(w, y)
        BaseGM_Error.__init__(self, y, x, w, constant=constant) 
        self.title = "SPATIALLY WEIGHTED LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_x.append('lambda')
        #### we currently ignore nonspat_diag parameter ####
        self._get_diagnostics(w=w, beta_diag=True, nonspat_diag=False,\
                                    vm=vm, pred=pred)

    def _get_diagnostics(self, beta_diag=True, w=None, nonspat_diag=True,\
                              vm=False, pred=False):
        USER.DiagnosticBuilder.__init__(self, w=w, beta_diag=True,\
                                            nonspat_diag=nonspat_diag,\
                                            vm=vm, pred=pred, instruments=False)

class BaseGM_Endog_Error(RegressionProps):
    '''
    Generalized Spatial Two Stages Least Squares (TSLS + GMM) using spatial
    error from Kelejian and Prucha (1998) [1]_ and Kelejian and Prucha (1999) [2]_
    ...

    Parameters
    ----------
    y           : array
                  nx1 array of dependent variable
    x           : array
                  nxk array of independent variables (assumed to be aligned with y)
    w           : W
                  Spatial weights instance 
    yend        : array
                  endogenous variables
    q           : array
                  array of external exogenous variables to use as instruments;
                  (note: this should not contain any variables from x; all x
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)

    Attributes
    ----------

    betas       : array
                  (k+1)x1 array with estimated coefficients (betas + lambda)
    se_betas    : array
                  kx1 array with standard errors for estimated coefficients
    pvals       : array
                  kx1 array with p-values of the estimated coefficients
    u           : array
                  Vector of residuals (Note it employs original x and y
                  instead of the spatially filtered ones)
    vm          : array
                  Variance-covariance matrix


    References
    ----------

    .. [1] Kelejian, H.R., Prucha, I.R. (1998) "A generalized spatial
    two-stage least squares procedure for estimating a spatial autoregressive
    model with autoregressive disturbances". The Journal of Real State
    Finance and Economics, 17, 1.

    .. [2] Kelejian, H.R., Prucha, I.R. (1999) "A Generalized Moments
    Estimator for the Autoregressive Parameter in a Spatial Model".
    International Economic Review, 40, 2.

    Examples
    --------

    >>> import pysal
    >>> import numpy as np
    >>> dbf = pysal.open('examples/columbus.dbf','r')
    >>> y = np.array([dbf.by_col('CRIME')]).T
    >>> x = np.array([dbf.by_col('INC')]).T
    >>> yend = np.array([dbf.by_col('HOVAL')]).T
    >>> q = np.array([dbf.by_col('DISCBD')]).T
    >>> w = pysal.open('examples/columbus.gal', 'r').read() 
    >>> w.transform='r'
    >>> model = BaseGM_Endog_Error(y, x, w, yend, q)
    >>> np.around(model.betas, decimals=6)
    array([[ 82.57298 ],
           [  0.580959],
           [ -1.448077],
           [  0.349917]])
    >>> np.around(model.se_betas, decimals=6)
    array([[ 16.138089],
           [  1.354476],
           [  0.786205]])

    '''
    def __init__(self, y, x, w, yend, q, constant=True):

        #1a. TSLS --> \tilde{betas}
        tsls = TSLS.BaseTSLS(y, x, yend, q=q, constant=constant)
        self.n, self.k = tsls.x.shape
        self.x = tsls.x
        self.y = tsls.y
        self.yend = tsls.yend

        #1b. GMM --> \tilde{\lambda1}
        moments = _momentsGM_Error(w, tsls.u)
        lambda1 = optim_moments(moments)

        #2a. 2SLS -->\hat{betas}
        xs = get_spFilter(w, lambda1, self.x)
        ys = get_spFilter(w, lambda1, self.y)
        yend_s = get_spFilter(w, lambda1, self.yend)
        tsls2 = TSLS.BaseTSLS(ys, xs, yend_s, h=tsls.h, constant=False)

        #Output
        self.betas = np.vstack((tsls2.betas, np.array([[lambda1]])))
        self.u = y - np.dot(tsls.z, tsls2.betas)
        sig2 = np.dot(tsls2.u.T,tsls2.u) / self.n
        self.vm = sig2 * tsls2.varb 
        self.se_betas = np.sqrt(self.vm.diagonal()).reshape(tsls2.betas.shape)
        zs = tsls2.betas / self.se_betas
        self.pvals = norm.sf(abs(zs)) * 2.
        self._cache = {}


class GM_Endog_Error(BaseGM_Endog_Error, USER.DiagnosticBuilder):
    '''


    Examples
    --------

    >>> import pysal
    >>> import numpy as np
    >>> dbf = pysal.open('examples/columbus.dbf','r')
    >>> y = np.array([dbf.by_col('CRIME')]).T
    >>> x = np.array([dbf.by_col('INC')]).T
    >>> yend = np.array([dbf.by_col('HOVAL')]).T
    >>> q = np.array([dbf.by_col('DISCBD')]).T
    >>> w = pysal.open('examples/columbus.gal', 'r').read() 
    >>> w.transform='r'
    >>> model = GM_Endog_Error(y, x, w, yend, q, name_x=['inc'], name_y='crime', name_yend=['hoval'], name_q=['discbd'], name_ds='columbus')
    >>> print model.name_z
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> np.around(model.betas, decimals=6)
    array([[ 82.57298 ],
           [  0.580959],
           [ -1.448077],
           [  0.349917]])
    >>> np.around(model.se_betas, decimals=6)
    array([[ 16.138089],
           [  1.354476],
           [  0.786205]])
    
    '''
    def __init__(self, y, x, w, yend, q, constant=True, nonspat_diag=True,\
                    name_y=None, name_x=None,\
                    name_yend=None, name_q=None, name_ds=None,\
                    vm=False, pred=False):        
        #### we currently ignore nonspat_diag parameter ####

        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        BaseGM_Endog_Error.__init__(self, y, x, w, yend, q, constant=constant)
        self.title = "GENERALIZED SPATIAL TWO STAGE LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')
        self.name_q = USER.set_name_q(name_q, q)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        #### we currently ignore nonspat_diag parameter ####
        self._get_diagnostics(w=w, beta_diag=True, nonspat_diag=False,\
                                    vm=vm, pred=pred)
        
    def _get_diagnostics(self, beta_diag=True, w=None, nonspat_diag=True,\
                              vm=False, pred=False):
        USER.DiagnosticBuilder.__init__(self, w=w, beta_diag=True,\
                                            nonspat_diag=nonspat_diag, lamb=True,\
                                            vm=vm, pred=pred, instruments=True)        

class BaseGM_Combo(BaseGM_Endog_Error, RegressionProps):
    """
    Generalized Spatial Two Stages Least Squares (TSLS + GMM) with spatial lag using spatial
    error from Kelejian and Prucha (1998) [1]_ and Kelejian and Prucha (1999) [2]_
    ...

    Parameters
    ----------

    y           : array
                  nx1 array with dependent variables
    x           : array
                  nxk array with independent variables aligned with y
    w           : W
                  PySAL weights instance aligned with y
    yend        : array
                  Optional. Additional non-spatial endogenous variables (spatial lag is added by default)
    q           : array
                  array of instruments for yend (note: this should not contain
                  any variables from x; spatial instruments are computed by 
                  default)
    w_lags      : int
                  Number of orders to power W when including it as intrument
                  for the spatial lag (e.g. if w_lags=1, then the only
                  instrument is WX; if w_lags=2, the instrument is WWX; and so
                  on)
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)

    Attributes
    ----------
    
    betas       : array
                  (k+1)x1 array with estimated coefficients (betas + lambda)
    se_betas    : array
                  kx1 array with standard errors for estimated coefficients
    pvals       : array
                  kx1 array with p-values of the estimated coefficients
    u           : array
                  Vector of residuals (Note it employs original x and y
                  instead of the spatially filtered ones)
    vm          : array
                  Variance-covariance matrix

    References
    ----------

    .. [1] Kelejian, H.R., Prucha, I.R. (1998) "A generalized spatial
    two-stage least squares procedure for estimating a spatial autoregressive
    model with autoregressive disturbances". The Journal of Real State
    Finance and Economics, 17, 1.

    .. [2] Kelejian, H.R., Prucha, I.R. (1999) "A Generalized Moments
    Estimator for the Autoregressive Parameter in a Spatial Model".
    International Economic Review, 40, 2.

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
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'

    Example only with spatial lag

    >>> reg = BaseGM_Combo(y, X, w)

    Print the betas

    >>> print np.around(np.hstack((reg.betas[:-1],np.sqrt(reg.vm.diagonal()).reshape(3,1))),3)
    [[ 39.06   11.86 ]
     [ -1.404   0.391]
     [  0.467   0.2  ]]
    

    And lambda

    >>> print 'Lamda: ', np.around(reg.betas[-1], 3)
    Lamda:  [-0.048]
        
    Example with both spatial lag and other endogenous variables

    >>> yd = []
    >>> yd.append(db.by_col("HOVAL"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> reg = BaseGM_Combo(y, X, w, yd, q)
    >>> betas = np.array([['Intercept'],['INC'],['HOVAL'],['W_CRIME']])
    >>> print np.hstack((betas, np.around(np.hstack((reg.betas[:-1], np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)))
    [['Intercept' '50.0944' '14.3593']
     ['INC' '-0.2552' '0.5667']
     ['HOVAL' '-0.6885' '0.3029']
     ['W_CRIME' '0.4375' '0.2314']]

        """
    def __init__(self, y, x, w, yend=None, q=None, w_lags=1,\
                    constant=True):
        # Create spatial lag of y
        yl = lag_spatial(w, y)
        if issubclass(type(yend), np.ndarray):  # spatial and non-spatial instruments
            lag_vars = np.hstack((x, q))
            spatial_inst = get_lags(w, lag_vars, w_lags)
            q_out = np.hstack((q, spatial_inst))
            yend_out = np.hstack((yend, yl))
        elif yend == None:                   # spatial instruments only
            q_out = get_lags(w, x, w_lags)
            yend_out = yl
        else:
            raise Exception, "invalid value passed to yend"
        BaseGM_Endog_Error.__init__(self, y, x, w, yend_out, q_out, constant=constant)

class GM_Combo(BaseGM_Combo, USER.DiagnosticBuilder):
    """


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
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'

    Example only with spatial lag

    >>> reg = GM_Combo(y, X, w, name_y='crime', name_x=['income'], name_ds='columbus')

    Print the betas

    >>> print reg.name_z
    ['CONSTANT', 'income', 'lag_crime', 'lambda']
    >>> print np.around(np.hstack((reg.betas[:-1],np.sqrt(reg.vm.diagonal()).reshape(3,1))),3)
    [[ 39.06   11.86 ]
     [ -1.404   0.391]
     [  0.467   0.2  ]]
    

    And lambda

    >>> print 'lambda: ', np.around(reg.betas[-1], 3)
    lambda:  [-0.048]
        
    Example with both spatial lag and other endogenous variables

    >>> yd = []
    >>> yd.append(db.by_col("HOVAL"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> reg = GM_Combo(y, X, w, yd, q, name_x=['inc'], name_y='crime', name_yend=['hoval'], name_q=['discbd'], name_ds='columbus')
    >>> print reg.name_z
    ['CONSTANT', 'inc', 'hoval', 'lag_crime', 'lambda']
    >>> names = np.array(reg.name_z).reshape(5,1)
    >>> print np.hstack((names[0:4,:], np.around(np.hstack((reg.betas[:-1], np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)))
    [['CONSTANT' '50.0944' '14.3593']
     ['inc' '-0.2552' '0.5667']
     ['hoval' '-0.6885' '0.3029']
     ['lag_crime' '0.4375' '0.2314']]

    >>> print 'lambda: ', np.around(reg.betas[-1], 3)
    lambda:  [ 0.254]
    """

    def __init__(self, y, x, w, yend=None, q=None, w_lags=1, constant=True,\
                    nonspat_diag=True, name_y=None, name_x=None, name_yend=None,\
                    name_q=None, name_ds=None,\
                    vm=False, pred=False):        
        #### we currently ignore nonspat_diag parameter ####

        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        BaseGM_Combo.__init__(self, y, x, w, yend, q, w_lags, constant)
        self.title = "GENERALIZED SPATIAL TWO STAGE LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_yend.append(USER.set_name_yend_sp(self.name_y))
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')
        self.name_q = USER.set_name_q(name_q, q)
        self.name_q.extend(USER.set_name_q_sp(self.name_x, w_lags))
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        #### we currently ignore nonspat_diag parameter ####
        self._get_diagnostics(w=w, beta_diag=True, nonspat_diag=False,\
                                    vm=vm, pred=pred)
     
    def _get_diagnostics(self, beta_diag=True, w=None, nonspat_diag=True,\
                              vm=False, pred=False):
        USER.DiagnosticBuilder.__init__(self, w=w, beta_diag=True,\
                                            nonspat_diag=nonspat_diag, lamb=True,\
                                            vm=vm, pred=pred, instruments=True)        

# Hom Models

class BaseGM_Error_Hom(RegressionProps):
    '''
    SWLS estimation of spatial error with homskedasticity. Based on 
    Anselin (2011) [1]_.
    ...

    Parameters
    ----------
    y           : array
                  nx1 array of dependent variable
    x           : array
                  nxk array of independent variables (assumed to be aligned with y)
    w           : W
                  Spatial weights instance 
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)
    A1          : str
                  Flag selecting the version of A1 to be used:
                    * 'hom'     : A1 for Hom as defined in Anselin 2011 (Default)
                    * 'hom_sc'  : A1 for Hom including scalar correctin as in Drukker
                    * 'het'     : A1 for Het


    Attributes
    ----------
    y           : array
                  nx1 array of dependent variable
    x           : array
                  array of independent variables (with constant added if
                  constant parameter set to True)
    betas       : array
                  (k+1)x1 array with estimates for betas and lambda
    u           : array
                  nx1 array of residuals 
    predy       : array
                  nx1 array of predicted values 
    n           : integer
                  number of observations
    k           : int
                  Number of variables, including exogenous and endogenous
                  variables and constant
    vm          : array
                  (k+1)x(k+1) variance-covariance matrix

    References
    ----------

    .. [1] Anselin, L. (2011) "GMM Estimation of Spatial Error Autocorrelation
    with and without Heteroskedasticity". 

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'

    Model commands

    >>> reg = BaseGM_Error_Hom(y, X, w, A1='hom_sc')
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 47.9479  12.3021]
     [  0.7063   0.4967]
     [ -0.556    0.179 ]
     [  0.4129   1.2844]]
    >>> print np.around(reg.vm, 4)
    [[  1.51340700e+02  -5.29060000e+00  -1.85650000e+00  -1.17200000e-01]
     [ -5.29060000e+00   2.46700000e-01   5.14000000e-02   1.56000000e-02]
     [ -1.85650000e+00   5.14000000e-02   3.21000000e-02  -2.90000000e-03]
     [ -1.17200000e-01   1.56000000e-02  -2.90000000e-03   1.64980000e+00]]
    '''
    def __init__(self, y, x, w, constant=True, A1='hom'):
        if A1 == 'hom':
            w.A1 = get_A1_hom(w.sparse)
        elif A1 == 'hom_sc':
            w.A1 = get_A1_hom(w.sparse, scalarKP=True)
        elif A1 == 'het':
            w.A1 = get_A1_het(w.sparse)

        w.A2 = get_A2_hom(w.sparse)

        # 1a. OLS --> \tilde{\delta}
        ols = OLS.BaseOLS(y, x, constant=constant)
        self.x, self.y, self.n, self.k = ols.x, ols.y, ols.n, ols.k

        # 1b. GM --> \tilde{\rho}
        moments = moments_hom(w, ols.u)
        lambda1 = optim_moments(moments)

        # 2a. SWLS --> \hat{\delta}
        x_s = get_spFilter(w,lambda1,self.x)
        y_s = get_spFilter(w,lambda1,self.y)
        ols_s = OLS.BaseOLS(y_s, x_s, constant=False)
        self.predy = np.dot(self.x, ols_s.betas)
        self.u = self.y - self.predy

        # 2b. GM 2nd iteration --> \hat{\rho}
        moments = moments_hom(w, self.u)
        psi = get_vc_hom(w, self, lambda1)[0]
        lambda2 = optim_moments(moments, psi)

        # Output
        self.betas = np.vstack((ols_s.betas,lambda2))
        self.vm = get_omega_hom_ols(w, self, lambda2, moments[0])
        self._cache = {}

class GM_Error_Hom(BaseGM_Error_Hom, USER.DiagnosticBuilder):
    '''
    User class for SWLS estimation of spatial error with homskedasticity.
    Based on Anselin (2011) [1]_.
    ...

    Parameters
    ----------
    y           : array
                  nx1 array of dependent variable
    x           : array
                  nxk array of independent variables (assumed to be aligned with y)
    w           : W
                  Spatial weights instance 
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)
    A1          : str
                  Flag selecting the version of A1 to be used:
                    * 'hom'     : A1 for Hom as defined in Anselin 2011 (Default)
                    * 'hom_sc'  : A1 for Hom including scalar correctin as in Drukker
                    * 'het'     : A1 for Het
    name_y      : string
                  Name of dependent variables for use in output
    name_x      : list of strings
                  Names of independent variables for use in output
    name_ds     : string
                  Name of dataset for use in output


    Attributes
    ----------
    y           : array
                  nx1 array of dependent variable
    x           : array
                  array of independent variables (with constant added if
                  constant parameter set to True)
    betas       : array
                  (k+1)x1 array with estimates for betas and lambda
    u           : array
                  nx1 array of residuals 
    predy       : array
                  nx1 array of predicted values 
    n           : integer
                  number of observations
    k           : int
                  Number of variables, including exogenous and endogenous
                  variables and constant
    vm          : array
                  (k+1)x(k+1) variance-covariance matrix

    References
    ----------

    .. [1] Anselin, L. (2011) "GMM Estimation of Spatial Error Autocorrelation
    with and without Heteroskedasticity". 

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X.append(db.by_col("CRIME"))
    >>> X = np.array(X).T
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'

    Model commands

    >>> reg = GM_Error_Hom(y, X, w, A1='hom_sc', name_y='home value', name_x=['income', 'crime'], name_ds='columbus')
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 47.9479  12.3021]
     [  0.7063   0.4967]
     [ -0.556    0.179 ]
     [  0.4129   1.2844]]


    '''
    def __init__(self, y, x, w, constant=True, A1='hom', nonspat_diag=True,\
                        name_y=None, name_x=None, name_ds=None,\
                        vm=False, pred=False):                
        #### we currently ignore nonspat_diag parameter ####

        USER.check_arrays(y, x)
        USER.check_weights(w, y)
        BaseGM_Error_Hom.__init__(self, y, x, w, constant=constant, A1=A1)
        self.title = "GENERALIZED SPATIAL LEAST SQUARES (Hom)"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_x.append('lambda')
        #### we currently ignore nonspat_diag parameter ####
        self._get_diagnostics(w=w, beta_diag=True, nonspat_diag=False,\
                                    vm=vm, pred=pred)

    def _get_diagnostics(self, beta_diag=True, w=None, nonspat_diag=True,\
                              vm=False, pred=False):
        USER.DiagnosticBuilder.__init__(self, w=w, beta_diag=True,\
                                            nonspat_diag=nonspat_diag,\
                                            vm=vm, pred=pred, instruments=False)


class BaseGM_Endog_Error_Hom(RegressionProps):
    '''
    Two step estimation of spatial error with endogenous regressors. Based on
    Drukker et al. (2010) [1]_ and Drukker et al. (2011) [2]_, following
    Anselin (2011) [3]_.
    ...

    Parameters
    ----------
    y           : array
                  nx1 array of dependent variable
    x           : array
                  nxk array of independent variables (assumed to be aligned with y)
    w           : W
                  Spatial weights instance 
    yend        : array
                  endogenous variables
    q           : array
                  array of external exogenous variables to use as instruments;
                  (note: this should not contain any variables from x; all x
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)
    A1          : str
                  Flag selecting the version of A1 to be used:
                    * 'hom'     : A1 for Hom as defined in Anselin 2011 (Default)
                    * 'hom_sc'  : A1 for Hom including scalar correctin as in Drukker
                    * 'het'     : A1 for Het


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
                  (k+1)x1 array with estimates for betas and lambda
    u           : array
                  nx1 array of residuals 
    predy       : array
                  nx1 array of predicted values 
    n           : integer
                  number of observations
    k           : int
                  Number of variables, including exogenous and endogenous
                  variables and constant
    vm          : array
                  (k+1)x(k+1) variance-covariance matrix

    References
    ----------

    .. [1] Drukker, D. M., Egger, P., Prucha, I. R. (2010) "On Two-step
    Estimation of a Spatial Autoregressive Model with Autoregressive
    Disturbances and Endogenous Regressors". Working paper.

    .. [2] Drukker, Prucha, I. R., Raciborski, R. (2010) "A command for
    estimating spatial-autoregressive models with spatial-autoregressive
    disturbances and additional endogenous variables". The Stata Journal, 1,
    N. 1, pp. 1-13.

    .. [3] Anselin, L. (2011) "GMM Estimation of Spatial Error Autocorrelation
    with and without Heteroskedasticity". 

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T
    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'
    >>> reg = BaseGM_Endog_Error_Hom(y, X, w, yd, q, A1='hom_sc')
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 55.3658  23.496 ]
     [  0.4643   0.7382]
     [ -0.669    0.3943]
     [  0.4321   0.1927]]

    
    '''
    def __init__(self, y, x, w, yend, q, constant=True, A1='hom'):

        if A1 == 'hom':
            w.A1 = get_A1_hom(w.sparse)
        elif A1 == 'hom_sc':
            w.A1 = get_A1_hom(w.sparse, scalarKP=True)
        elif A1 == 'het':
            w.A1 = get_A1_het(w.sparse)

        w.A2 = get_A2_hom(w.sparse)

        # 1a. S2SLS --> \tilde{\delta}
        tsls = TSLS.BaseTSLS(y, x, yend, q=q, constant=constant)
        self.x, self.z, self.h, self.y = tsls.x, tsls.z, tsls.h, tsls.y
        self.yend, self.q, self.n, self.k = tsls.yend, tsls.q, tsls.n, tsls.k

        # 1b. GM --> \tilde{\rho}
        moments = moments_hom(w, tsls.u)
        lambda1 = optim_moments(moments)

        # 2a. GS2SLS --> \hat{\delta}
        x_s = get_spFilter(w,lambda1,self.x)
        y_s = get_spFilter(w,lambda1,self.y)
        yend_s = get_spFilter(w, lambda1, self.yend)
        tsls_s = TSLS.BaseTSLS(y_s, x_s, yend_s, h=self.h, constant=False)
        predy = np.dot(self.z, tsls_s.betas)
        self.u = self.y - predy
        self.predy = predy

        # 2b. GM 2nd iteration --> \hat{\rho}
        moments = moments_hom(w, self.u)
        psi = get_vc_hom(w, self, lambda1, tsls_s.z)[0]
        lambda2 = optim_moments(moments, psi)

        # Output
        self.betas = np.vstack((tsls_s.betas,lambda2))
        self.vm = get_omega_hom(w, self, lambda2, moments[0])
        self._cache = {}

class GM_Endog_Error_Hom(BaseGM_Endog_Error_Hom, USER.DiagnosticBuilder):
    '''
    User clasee for two step estimation of spatial error with endogenous
    regressors. Based on Drukker et al. (2010) [1]_ and Drukker et al. (2011)
    [2]_, following Anselin (2011) [3]_.
    ...

    Parameters
    ----------
    y           : array
                  nx1 array of dependent variable
    x           : array
                  nxk array of independent variables (assumed to be aligned with y)
    w           : W
                  Spatial weights instance 
    yend        : array
                  endogenous variables
    q           : array
                  array of external exogenous variables to use as instruments;
                  (note: this should not contain any variables from x; all x
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)
    A1          : str
                  Flag selecting the version of A1 to be used:
                    * 'hom'     : A1 for Hom as defined in Anselin 2011 (Default)
                    * 'hom_sc'  : A1 for Hom including scalar correctin as in Drukker
                    * 'het'     : A1 for Het
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
                  (k+1)x1 array with estimates for betas and lambda
    u           : array
                  nx1 array of residuals 
    predy       : array
                  nx1 array of predicted values 
    n           : integer
                  number of observations
    k           : int
                  Number of variables, including exogenous and endogenous
                  variables and constant
    vm          : array
                  (k+1)x(k+1) variance-covariance matrix

    References
    ----------

    .. [1] Drukker, D. M., Egger, P., Prucha, I. R. (2010) "On Two-step
    Estimation of a Spatial Autoregressive Model with Autoregressive
    Disturbances and Endogenous Regressors". Working paper.

    .. [2] Drukker, Prucha, I. R., Raciborski, R. (2010) "A command for
    estimating spatial-autoregressive models with spatial-autoregressive
    disturbances and additional endogenous variables". The Stata Journal, 1,
    N. 1, pp. 1-13.

    .. [3] Anselin, L. (2011) "GMM Estimation of Spatial Error Autocorrelation
    with and without Heteroskedasticity". 

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T
    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'

    Model commands

    >>> reg = GM_Endog_Error_Hom(y, X, w, yd, q, A1='hom_sc', name_x=['inc'], name_y='hoval', name_yend=['crime'], name_q=['discbd'], name_ds='columbus')
    >>> print reg.name_z
    ['CONSTANT', 'inc', 'crime', 'lambda']
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 55.3658  23.496 ]
     [  0.4643   0.7382]
     [ -0.669    0.3943]
     [  0.4321   0.1927]]

        '''
    def __init__(self, y, x, w, yend, q, constant=True, A1='hom',\
                    nonspat_diag=True, name_y=None, name_x=None,\
                    name_yend=None, name_q=None, name_ds=None,\
                    vm=False, pred=False):        
        #### we currently ignore nonspat_diag parameter ####

        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        BaseGM_Endog_Error_Hom.__init__(self, y, x, w, yend, q, constant=constant, A1=A1)
        self.title = "GENERALIZED SPATIAL TWO STAGE LEAST SQUARES (Hom)"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')  #listing lambda last
        self.name_q = USER.set_name_q(name_q, q)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        #### we currently ignore nonspat_diag parameter ####
        self._get_diagnostics(w=w, beta_diag=True, nonspat_diag=False,\
                                    vm=vm, pred=pred)
        
    def _get_diagnostics(self, beta_diag=True, w=None, nonspat_diag=True,\
                              vm=False, pred=False):
        USER.DiagnosticBuilder.__init__(self, w=w, beta_diag=True,\
                                            nonspat_diag=nonspat_diag, lamb=True,\
                                            vm=vm, pred=pred, instruments=True)        


class BaseGM_Combo_Hom(BaseGM_Endog_Error_Hom, RegressionProps):
    '''
    Two step estimation of spatial lag and spatial error with endogenous
    regressors. Based on Drukker et al. (2010) [1]_ and Drukker et al. (2011) [2]_,
    following Anselin (2011) [3]_.
    ...

    Parameters
    ----------
    y           : array
                  nx1 array with dependent variable
    x           : array
                  nxk array with independent variables aligned with y
    w           : W
                  PySAL weights instance aligned with y
    yend        : array
                  Optional. Additional non-spatial endogenous variables (spatial lag is added by default)
    q           : array
                  array of instruments for yend (note: this should not contain
                  any variables from x; spatial instruments are computed by 
                  default)
    w_lags      : int
                  Number of orders to power W when including it as intrument
                  for the spatial lag (e.g. if w_lags=1, then the only
                  instrument is WX; if w_lags=2, the instrument is WWX; and so
                  on)    
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)
    A1          : str
                  Flag selecting the version of A1 to be used:
                    * 'hom'     : A1 for Hom as defined in Anselin 2011 (Default)
                    * 'hom_sc'  : A1 for Hom including scalar correctin as in Drukker
                    * 'het'     : A1 for Het

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
                  (k+1)x1 array with estimates for betas and lambda
    u           : array
                  nx1 array of residuals 
    predy       : array
                  nx1 array of predicted values 
    n           : integer
                  number of observations
    k           : int
                  Number of variables, including exogenous and endogenous
                  variables and constant
    vm          : array
                  (k+1)x(k+1) variance-covariance matrix

    References
    ----------

    .. [1] Drukker, D. M., Egger, P., Prucha, I. R. (2010) "On Two-step
    Estimation of a Spatial Autoregressive Model with Autoregressive
    Disturbances and Endogenous Regressors". Working paper.

    .. [2] Drukker, Prucha, I. R., Raciborski, R. (2010) "A command for
    estimating spatial-autoregressive models with spatial-autoregressive
    disturbances and additional endogenous variables". The Stata Journal, 1,
    N. 1, pp. 1-13.

    .. [3] Anselin, L. (2011) "GMM Estimation of Spatial Error Autocorrelation
    with and without Heteroskedasticity". 

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'

    Example only with spatial lag

    >>> reg = BaseGM_Combo_Hom(y, X, w, A1='hom_sc')
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 10.1254  15.2869]
     [  1.5683   0.4407]
     [  0.1513   0.4048]
     [  0.2103   0.4226]]


    Example with both spatial lag and other endogenous variables

    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> reg = BaseGM_Combo_Hom(y, X, w, yd, q, A1='hom_sc')
    >>> betas = np.array([['CONSTANT'],['inc'],['crime'],['lag_hoval'],['lambda']])
    >>> print np.hstack((betas, np.around(np.hstack((reg.betas, np.sqrt(reg.vm.diagonal()).reshape(5,1))),5)))
    [['CONSTANT' '111.77058' '67.75192']
     ['inc' '-0.30974' '1.16656']
     ['crime' '-1.36043' '0.6841']
     ['lag_hoval' '-0.52908' '0.84428']
     ['lambda' '0.60116' '0.18605']]

    '''
    def __init__(self, y, x, w, yend=None, q=None, w_lags=1,\
                    constant=True, A1='hom'):
        # Create spatial lag of y
        yl = lag_spatial(w, y)
        if issubclass(type(yend), np.ndarray):  # spatial and non-spatial instruments
            lag_vars = np.hstack((x, q))
            spatial_inst = get_lags(w ,lag_vars, w_lags)
            q = np.hstack((q, spatial_inst))
            yend = np.hstack((yend, yl))
        elif yend == None:                   # spatial instruments only
            q = get_lags(w, x, w_lags)
            yend = yl
        else:
            raise Exception, "invalid value passed to yend"
        BaseGM_Endog_Error_Hom.__init__(self, y, x, w, yend, q, A1=A1)

class GM_Combo_Hom(BaseGM_Combo_Hom, USER.DiagnosticBuilder):
    '''
    Two step estimation of spatial lag and spatial error with endogenous
    regressors. Based on Drukker et al. (2010) [1]_ and Drukker et al. (2011) [2]_,
    following Anselin (2011) [3]_.
    ...

    Parameters
    ----------
    y           : array
                  nx1 array with dependent variable
    x           : array
                  nxk array with independent variables aligned with y
    w           : W
                  PySAL weights instance aligned with y
    yend        : array
                  Optional. Additional non-spatial endogenous variables (spatial lag is added by default)
    q           : array
                  array of instruments for yend (note: this should not contain
                  any variables from x; spatial instruments are computed by 
                  default)
    w_lags      : int
                  Number of orders to power W when including it as intrument
                  for the spatial lag (e.g. if w_lags=1, then the only
                  instrument is WX; if w_lags=2, the instrument is WWX; and so
                  on)    
    constant    : boolean
                  If true it appends a vector of ones to the independent variables
                  to estimate intercept (set to True by default)
    A1          : str
                  Flag selecting the version of A1 to be used:
                    * 'hom'     : A1 for Hom as defined in Anselin 2011 (Default)
                    * 'hom_sc'  : A1 for Hom including scalar correctin as in Drukker
                    * 'het'     : A1 for Het
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
                  (k+1)x1 array with estimates for betas and lambda
    u           : array
                  nx1 array of residuals 
    predy       : array
                  nx1 array of predicted values 
    n           : integer
                  number of observations
    k           : int
                  Number of variables, including exogenous and endogenous
                  variables and constant
    vm          : array
                  (k+1)x(k+1) variance-covariance matrix

    References
    ----------

    .. [1] Drukker, D. M., Egger, P., Prucha, I. R. (2010) "On Two-step
    Estimation of a Spatial Autoregressive Model with Autoregressive
    Disturbances and Endogenous Regressors". Working paper.

    .. [2] Drukker, Prucha, I. R., Raciborski, R. (2010) "A command for
    estimating spatial-autoregressive models with spatial-autoregressive
    disturbances and additional endogenous variables". The Stata Journal, 1,
    N. 1, pp. 1-13.

    .. [3] Anselin, L. (2011) "GMM Estimation of Spatial Error Autocorrelation
    with and without Heteroskedasticity". 

    Examples
    --------
    >>> import numpy as np
    >>> import pysal
    >>> db=pysal.open("examples/columbus.dbf","r")
    >>> y = np.array(db.by_col("HOVAL"))
    >>> y = np.reshape(y, (49,1))
    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T
    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")
    >>> w.transform = 'r'

    Example only with spatial lag

    >>> reg = GM_Combo_Hom(y, X, w, A1='hom_sc', name_x=['inc'],\
            name_y='hoval', name_yend=['crime'], name_q=['discbd'],\
            name_ds='columbus')
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 10.1254  15.2869]
     [  1.5683   0.4407]
     [  0.1513   0.4048]
     [  0.2103   0.4226]]


    Example with both spatial lag and other endogenous variables

    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> reg = GM_Combo_Hom(y, X, w, yd, q, A1='hom_sc', \
            name_ds='columbus')
    >>> betas = np.array([['CONSTANT'],['inc'],['crime'],['lag_hoval'],['lambda']])
    >>> print np.hstack((betas, np.around(np.hstack((reg.betas, np.sqrt(reg.vm.diagonal()).reshape(5,1))),5)))
    [['CONSTANT' '111.77058' '67.75192']
     ['inc' '-0.30974' '1.16656']
     ['crime' '-1.36043' '0.6841']
     ['lag_hoval' '-0.52908' '0.84428']
     ['lambda' '0.60116' '0.18605']]

    '''
    def __init__(self, y, x, w, yend=None, q=None, w_lags=1,\
                    constant=True, A1='hom', nonspat_diag=True,\
                    name_y=None, name_x=None, name_yend=None,\
                    name_q=None, name_ds=None,\
                    vm=False, pred=False):        
        #### we currently ignore nonspat_diag parameter ####

        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        BaseGM_Combo_Hom.__init__(self, y, x, w, yend, q, w_lags,\
                constant, A1)
        self.title = "GENERALIZED SPATIAL TWO STAGE LEAST SQUARES (Hom)"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_yend.append(USER.set_name_yend_sp(self.name_y))
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')  #listing lambda last
        self.name_q = USER.set_name_q(name_q, q)
        self.name_q.extend(USER.set_name_q_sp(self.name_x, w_lags))
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        #### we currently ignore nonspat_diag parameter ####
        self._get_diagnostics(w=w, beta_diag=True, nonspat_diag=False,\
                                    vm=vm, pred=pred)
     
    def _get_diagnostics(self, beta_diag=True, w=None, nonspat_diag=True,\
                              vm=False, pred=False):
        USER.DiagnosticBuilder.__init__(self, w=w, beta_diag=True,\
                                            nonspat_diag=nonspat_diag, lamb=True,\
                                            vm=vm, pred=pred, instruments=True)        


def moments_hom(w, u):
    '''
    Compute G and g matrices for the spatial error model with homoscedasticity
    as in Anselin [1]_ (2011).
    ...

    Parameters
    ----------

    w           : W
                  Spatial weights instance

    u           : array
                  Residuals. nx1 array assumed to be aligned with w
 
    Attributes
    ----------

    moments     : list
                  List of two arrays corresponding to the matrices 'G' and
                  'g', respectively.


    References
    ----------

    .. [1] Anselin, L. (2011) "GMM Estimation of Spatial Error Autocorrelation
    with and without Heteroskedasticity". 
    '''
    n = w.sparse.shape[0]
    A1u = w.A1 * u
    A2u = w.A2 * u
    wu = w.sparse * u

    g1 = np.dot(u.T, A1u)
    g2 = np.dot(u.T, A2u)
    g = np.array([[g1][0][0],[g2][0][0]]) / n

    G11 = 2 * np.dot(wu.T * w.A1, u)
    G12 = -np.dot(wu.T * w.A1, wu)
    G21 = 2 * np.dot(wu.T * w.A2, u)
    G22 = -np.dot(wu.T * w.A2, wu)
    G = np.array([[G11[0][0],G12[0][0]],[G21[0][0],G22[0][0]]]) / n
    return [G, g]

def get_vc_hom(w, reg, lambdapar, z_s=None, for_omegaOLS=False):
    '''
    VC matrix \psi of Spatial error with homoscedasticity. As in 
    Anselin (2011) [1]_ (p. 20)
    ...

    Parameters
    ----------
    w               :   W
                        Weights with A1 appended
    reg             :   reg
                        Regression object
    lambdapar       :   float
                        Spatial parameter estimated in previous step of the
                        procedure
    z_s             :   array
                        optional argument for spatially filtered Z (to be
                        passed only if endogenous variables are present)
    for_omegaOLS    :   boolean
                        If True (default=False), it also returns P, needed
                        only in the computation of Omega

    Returns
    -------

    psi         : array
                  2x2 VC matrix
    a1          : array
                  nx1 vector a1. If z_s=None, a1 = 0.
    a2          : array
                  nx1 vector a2. If z_s=None, a2 = 0.
    p           : array
                  P matrix. If z_s=None or for_omegaOLS=False, p=0.

    References
    ----------

    .. [1] Anselin, L. (2011) "GMM Estimation of Spatial Error Autocorrelation
    with and without Heteroskedasticity". 

    '''
    u_s = get_spFilter(w, lambdapar, reg.u)
    n = float(w.n)
    sig2 = np.dot(u_s.T, u_s) / n
    mu3 = np.sum(u_s**3) / n
    mu4 = np.sum(u_s**4) / n

    tr11 = w.A1 * w.A1
    tr11 = np.sum(tr11.diagonal())
    tr12 = w.A1 * (w.A2 * 2)
    tr12 = np.sum(tr12.diagonal())
    tr22 = w.A2 * w.A2 * 2
    tr22 = np.sum(tr22.diagonal())
    vecd1 = np.array([w.A1.diagonal()]).T

    psi11 = 2 * sig2**2 * tr11 + \
            (mu4 - 3 * sig2**2) * np.dot(vecd1.T, vecd1)
    psi12 = sig2**2 * tr12
    psi22 = sig2**2 * tr22

    a1, a2, p = 0., 0., 0.

    if for_omegaOLS:
        x_s = get_spFilter(w, lambdapar, reg.x)
        p = la.inv(np.dot(x_s.T, x_s) / n)

    if issubclass(type(z_s), np.ndarray):
        alpha1 = (-2/n) * np.dot(z_s.T, w.A1 * u_s)
        alpha2 = (-2/n) * np.dot(z_s.T, w.A2 * u_s)

        hth = np.dot(reg.h.T, reg.h)
        hthni = la.inv(hth / n) 
        htzsn = np.dot(reg.h.T, z_s) / n 
        p = np.dot(hthni, htzsn)
        p = np.dot(p, la.inv(np.dot(htzsn.T, p)))
        hp = np.dot(reg.h, p)
        a1 = np.dot(hp, alpha1)
        a2 = np.dot(hp, alpha2)

        psi11 = psi11 + \
            sig2 * np.dot(a1.T, a1) + \
            2 * mu3 * np.dot(a1.T, vecd1)
        psi12 = psi12 + \
            sig2 * np.dot(a1.T, a2) + \
            mu3 * np.dot(a2.T, vecd1) # 3rd term=0
        psi22 = psi22 + \
            sig2 * np.dot(a2.T, a2) # 3rd&4th terms=0 bc vecd2=0

    psi = np.array([[psi11[0][0], psi12[0][0]], [psi12[0][0], psi22[0][0]]]) / n
    return psi, a1, a2, p

def get_omega_hom(w, reg, lamb, G):
    '''
    Omega VC matrix for Hom models with endogenous variables computed as in
    Anselin (2011) [1]_ (p. 21).
    ...

    Parameters
    ----------
    w       :   W
                Weights with A1 appended
    reg     :   reg
                Regression object
    lamb    :   float
                Spatial parameter estimated in previous step of the
                procedure
    G       :   array
                Matrix 'G' of the moment equation

    Returns
    -------
    omega   :   array
                Omega matrix of VC of the model

    References
    ----------

    .. [1] Anselin, L. (2011) "GMM Estimation of Spatial Error Autocorrelation
    with and without Heteroskedasticity". 

    '''
    n = float(w.n)
    z_s = get_spFilter(w, lamb, reg.z)
    u_s = get_spFilter(w, lamb, reg.u)
    sig2 = np.dot(u_s.T, u_s) / n
    mu3 = np.sum(u_s**3) / n
    vecdA1 = np.array([w.A1.diagonal()]).T
    psi, a1, a2, p = get_vc_hom(w, reg, lamb, z_s)
    j = np.dot(G, np.array([[1.], [2*lamb]]))
    psii = la.inv(psi)
    psiDL = (mu3 * np.dot(reg.h.T, np.hstack((vecdA1, np.zeros((n, 1))))) + \
            sig2 * np.dot(reg.h.T, np.hstack((a1, a2)))) / n

    oDD = np.dot(la.inv(np.dot(reg.h.T, reg.h)), np.dot(reg.h.T, z_s))
    oDD = sig2 * la.inv(np.dot(z_s.T, np.dot(reg.h, oDD)))
    oLL = la.inv(np.dot(j.T, np.dot(psii, j))) / n
    oDL = np.dot(np.dot(np.dot(p.T, psiDL), np.dot(psii, j)), oLL)

    o_upper = np.hstack((oDD, oDL))
    o_lower = np.hstack((oDL.T, oLL))
    return np.vstack((o_upper, o_lower))

def get_omega_hom_ols(w, reg, lamb, G):
    '''
    Omega VC matrix for Hom models without endogenous variables (OLS) computed
    as in Anselin (2011) [1]_.
    ...

    Parameters
    ----------
    w       :   W
                Weights with A1 appended
    reg     :   reg
                Regression object
    lamb    :   float
                Spatial parameter estimated in previous step of the
                procedure
    G       :   array
                Matrix 'G' of the moment equation

    Returns
    -------
    omega   :   array
                Omega matrix of VC of the model

    References
    ----------

    .. [1] Anselin, L. (2011) "GMM Estimation of Spatial Error Autocorrelation
    with and without Heteroskedasticity". 

    '''
    n = float(w.n)
    x_s = get_spFilter(w, lamb, reg.x)
    u_s = get_spFilter(w, lamb, reg.u)
    sig2 = np.dot(u_s.T, u_s) / n
    vecdA1 = np.array([w.A1.diagonal()]).T
    psi, a1, a2, p = get_vc_hom(w, reg, lamb, for_omegaOLS=True)
    j = np.dot(G, np.array([[1.], [2*lamb]]))
    psii = la.inv(psi)

    oDD = sig2 * la.inv(np.dot(x_s.T, x_s))
    oLL = la.inv(np.dot(j.T, np.dot(psii, j)))
    #oDL = np.zeros((oDD.shape[0], oLL.shape[1]))
    mu3 = np.sum(u_s**3) / n
    psiDL = (mu3 * np.dot(reg.x.T, np.hstack((vecdA1, np.zeros((n, 1)))))) / n
    oDL = np.dot(np.dot(np.dot(p.T, psiDL), np.dot(psii, j)), oLL)

    o_upper = np.hstack((oDD, oDL))
    o_lower = np.hstack((oDL.T, oLL))
    return np.vstack((o_upper, o_lower))

def get_omega_hom_deprecated(w, reg, lamb, G):
    '''
    NOTE: this implements full structure for 2SLS models at the end of p. 20
    in Luc's notes, not the reduced form at the end of p. 21.

    VC matrix \Omega of Spatial error with homoscedasticity. As in p. 11 of
    Drukker et al. (2011) [1]_

    To Do:
        * Optimize (pieces can be passed in instead of recomputed
    ...

    Parameters
    ----------

    w           : W
                  Spatial weights instance 

    lamb        : float
                  Spatial autoregressive parameter
                  
    reg         : reg
                  Regression object
    G           : array
                  Matrix G in moments equation
    psi         : array
                  Weighting matrix
 
    Returns
    -------

    omega       : array
                  (k+s)x(k+s) where s is the number of spatial parameters,
                  either one or two.

    References
    ----------

    .. [1] Drukker, Prucha, I. R., Raciborski, R. (2010) "A command for
    estimating spatial-autoregressive models with spatial-autoregressive
    disturbances and additional endogenous variables". The Stata Journal, 1,
    N. 1, pp. 1-13.

    '''
    n = float(w.n)
    z_s = get_spFilter(w, lamb, reg.z)
    psi, a1, a2, p = get_vc_hom(w, reg, lamb, z_s)
    u_s = get_spFilter(w, lamb, reg.u)
    sig2 = np.dot(u_s.T, u_s) / n
    mu3 = np.sum(u_s**3) / n
    vecdA1 = np.array([w.A1.diagonal()]).T

    psiDD = sig2 * np.dot(reg.h.T, reg.h) / n
    psiDL = (mu3 * np.dot(reg.h.T, np.hstack((vecdA1, np.zeros((n, 1))))) + \
            sig2 * np.dot(reg.h.T, np.hstack((a1, a2)))) / n
    psiO_upper = np.hstack((psiDD, psiDL))
    psiO_lower = np.hstack((psiDL.T, psi))
    psiO = np.vstack((psiO_upper, psiO_lower))

    j = np.dot(G, np.array([[1.], [2*lamb]]))
    psii = la.inv(psi)

    oL = la.inv(np.dot(np.dot(j.T, psii), j))
    oL = np.dot(oL, np.dot(j.T, psii))
    oL_upper = np.hstack((p.T, np.zeros((p.T.shape[0], oL.shape[1]))))
    oL_lower = np.hstack((np.zeros((oL.shape[0], p.T.shape[1])), oL))
    oL = np.vstack((oL_upper, oL_lower))

    oR = np.dot(np.dot(psii, j), la.inv(np.dot(np.dot(j.T, psii), j)))
    oR_upper = np.hstack((p, np.zeros((p.shape[0], oR.shape[1]))))
    oR_lower = np.hstack((np.zeros((oR.shape[0], p.shape[1])), oR))
    oR = np.vstack((oR_upper, oR_lower))

    return np.dot(np.dot(oL, psiO), oR)

def _get_a1a2_filt(w, reg, lambdapar, apat, wpwt, e, z_s):
    '''
    [DEPRECATED]
    Internal helper function to compute a1 and a2 in get_vc_hom. It assumes
    residuals come from a spatially filetered model
    '''
    alpha1 = np.dot(z_s.T, apat * e) / -w.n
    alpha2 = np.dot(z_s.T, wpwt * e) / -w.n

    q_hh = reg.hth / w.n
    q_hhi = la.inv(q_hh)
    q_hzs = np.dot(reg.h.T, z_s) / w.n
    p_s = np.dot(q_hhi, q_hzs)
    p_s = np.dot(p_s, la.inv(np.dot(q_hzs.T, np.dot(q_hhi, q_hzs.T))))
    t = np.dot(reg.h, p_s)
    return np.dot(t, alpha1), np.dot(t, alpha2), p_s

def __get_a1a2(w,reg,lambdapar):
    '''
    Method borrowed from spHetError. It computes a1, a2 as in section 4.3.2 of
    Luc's notes. It assumes residuals come from an original model.
    '''
    zst = get_spFilter(w,lambdapar, reg.z).T
    us = get_spFilter(w,lambdapar, reg.u)
    alpha1 = (-2.0/w.n) * (np.dot((zst * w.A1), us))
    alpha2 = (-1.0/w.n) * (np.dot((zst * (w.sparse + w.sparse.T)), us))
    v1 = np.dot(np.dot(reg.h, reg.pfora1a2), alpha1)
    v2 = np.dot(np.dot(reg.h, reg.pfora1a2), alpha2)
    a1t = power_expansion(w, v1, lambdapar, transpose=True)
    a2t = power_expansion(w, v2, lambdapar, transpose=True)
    return [a1t.T, a2t.T, reg.pfora1a2]

def _get_traces(A1, s):
    '''
    Parallel computation for traces in vm_hom
    '''
    

def _inference(ols):
    """
    DEPRECATED: not in current use

    Inference for estimated coefficients
    Coded as in GMerrorsar from R (which matches) using se_betas from diagnostics module
    """
    c = np.sqrt((ols.n-ols.k) / float(ols.n))
    ses = np.array([se_betas(ols) * c]).T
    zs = ols.betas / ses
    pvals = norm.sf(abs(zs)) * 2.
    return [ses, zs, pvals]

def _momentsGM_Error(w, u):

    u2 = np.dot(u.T, u)
    wu = w.sparse * u
    uwu = np.dot(u.T, wu)
    wu2 = np.dot(wu.T, wu)
    wwu = w.sparse * wu
    uwwu = np.dot(u.T, wwu)
    wwu2 = np.dot(wwu.T, wwu)
    wuwwu = np.dot(wu.T, wwu)
    wtw = w.sparse.T * w.sparse
    trWtW = np.sum(wtw.diagonal())

    g = np.array([[u2[0][0], wu2[0][0], uwu[0][0]]]).T / w.n

    G = np.array([[2 * uwu[0][0], -wu2[0][0], w.n], [2 * wuwwu[0][0], -wwu2[0][0], trWtW], [uwwu[0][0] + wu2[0][0], -wuwwu[0][0], 0.]]) / w.n

    return [G, g]

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':

    _test()

    """
    import numpy as np
    import pysal
    db=pysal.open("examples/columbus.dbf","r")
    y = np.array(db.by_col("HOVAL"))
    y = np.reshape(y, (49,1))
    X = []
    X.append(db.by_col("INC"))
    X.append(db.by_col("CRIME"))
    X = np.array(X).T
    w = pysal.rook_from_shapefile("examples/columbus.shp")
    w = pysal.open('examples/columbus.gal', 'r').read()    
    w.transform = 'r'
    q = []
    q.append(db.by_col("DISCBD"))
    q = np.array(q).T
    yd = []
    yd.append(db.by_col("CRIME"))
    yd = np.array(yd).T

    #model = BaseGM_Error_Hom(y, X, w, A1='hom') 
    ones = np.ones(y.shape)
    model = BaseGM_Endog_Error_Hom(y, ones, w, yend=X, q=X, constant=False, A1='hom')

    #model = BaseGM_Endog_Error_Hom(y, X, w, yend=yd, q=q, A1='hom_sc') #MATCHES
    #model = BaseGM_Combo_Hom(y, X, w, A1='hom_sc', w_lags=2) #MATCHES
    #model = BaseGM_Combo_Hom(y, X, w, yend=yd, q=q, A1='hom_sc', w_lags=2) #MATCHES
    print '\n'
    print np.around(np.hstack((model.betas,np.sqrt(model.vm.diagonal()).reshape(model.betas.shape[0],1))),8)
    print '\n'
    for row in model.vm:
        print map(np.round, row, [5]*len(row))

    tsls = TSLS.BaseTSLS(y, x, yd, q=q, constant=True)
    print tsls.betas
    psi = get_vc_hom(w, tsls, 0.3)
    print psi
    print '\n\tGM_Error model Example'
    model = GM_Error(x, y, w)
    print '\n### Betas ###'
    print model.betas
    print '\n### Std. Errors ###'
    print model.se_betas
    print '\n### z-Values ###'
    print model.z
    print '\n### p-Values ###'
    print model.pvals
    print '\n### Lambda ###'
    print model.lamb
    print '\n### Sig2 ###'
    print model.sig2
    """
