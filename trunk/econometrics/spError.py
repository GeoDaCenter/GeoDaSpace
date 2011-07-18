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
from utils import get_A1_hom, get_A1_het, optim_moments, get_spFilter, get_lags, _moments2eqs
import twosls as TSLS
import pysal.spreg.user_output as USER



class BaseGM_Error:
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
        w.A1 = get_A1_het(w.sparse)   #LA why?

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

        self.se_betas, self.z, self.pvals = _inference(ols2)
        self.step2OLS = ols2

class GM_Error(BaseGM_Error):
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
    >>> model = GM_Error(y, x, w, name_y='hoval', name_x=['income', 'crime'], name_ds='columbus')
    >>> print model.name_x
    ['CONSTANT', 'income', 'crime', 'lambda']
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
    def __init__(self, y, x, w, constant=True, name_y=None,\
                        name_x=None, name_ds=None):
        USER.check_arrays(y, x)
        USER.check_weights(w, y)
        BaseGM_Error.__init__(self, y, x, w, constant=constant) 
        self.title = "SPATIALLY WEIGHTED LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(copy.copy(name_x), x, constant)
        self.name_x.append('lambda')
        self.summary = "results place holder"


class BaseGM_Endog_Error:
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
    array([[ 9.90073 ],
           [ 0.667087],
           [ 0.190239]])
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
        self.vm = sig2 * la.inv(np.dot(tsls2.z.T,tsls2.z))  #LA should be tsls2.varb not Z'Z
        self.se_betas = np.sqrt(self.vm.diagonal()).reshape(tsls2.betas.shape)
        zs = tsls2.betas / self.se_betas
        self.pvals = norm.sf(abs(zs)) * 2.


class GM_Endog_Error(BaseGM_Endog_Error):
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
    array([[ 9.90073 ],
           [ 0.667087],
           [ 0.190239]])
    '''
    def __init__(self, y, x, w, yend, q, constant=True, name_y=None,\
                        name_x=None, name_yend=None, name_q=None,\
                        name_ds=None):
        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        BaseGM_Endog_Error.__init__(self, y, x, w, yend, q, constant=constant)
        self.title = "GENERALIZED SPATIAL STAGE LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')
        self.name_q = USER.set_name_q(name_q, q)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.summary = "results place holder"
        

class BaseGM_Combo(BaseGM_Endog_Error):
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
    [[ 39.06    9.533]
     [ -1.404   0.345]
     [  0.467   0.155]]

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
    [['Intercept' '50.0944' '10.7064']
     ['INC' '-0.2552' '0.4064']
     ['HOVAL' '-0.6885' '0.1079']
     ['W_CRIME' '0.4375' '0.1912']]
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

class GM_Combo(BaseGM_Combo):
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
    [[ 39.06    9.533]
     [ -1.404   0.345]
     [  0.467   0.155]]

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
    [['CONSTANT' '50.0944' '10.7064']
     ['inc' '-0.2552' '0.4064']
     ['hoval' '-0.6885' '0.1079']
     ['lag_crime' '0.4375' '0.1912']]
    >>> print 'lambda: ', np.around(reg.betas[-1], 3)
    lambda:  [ 0.254]
    """

    def __init__(self, y, x, w, yend=None, q=None, w_lags=1, constant=True,\
                    name_y=None, name_x=None, name_yend=None,\
                    name_q=None, name_ds=None):

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
        self.summary = "results place holder"


class BaseGM_Endog_Error_Hom:
    '''
    Two step estimation of spatial error with endogenous regressors. Based on 

    Based on Drukker et al. (2010) [1]_ and Drukker et al. (2011) [2]_
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
                  Array of beta coefficients, rho and lambda
    vm          : array
                  VC matrix Omega for beta coefficients, rho and lambda
    tsls        : reg
                  Regression object from initial two stage least squares
    lambda1     : float
                  Initial estimation of lambda (\tilde{\lambda})

    References
    ----------

    .. [1] Drukker, D. M., Egger, P., Prucha, I. R. (2010) "On Two-step
    Estimation of a Spatial Autoregressive Model with Autoregressive
    Disturbances and Endogenous Regressors". Working paper.

    .. [2] Drukker, Prucha, I. R., Raciborski, R. (2010) "A command for
    estimating spatial-autoregressive models with spatial-autoregressive
    disturbances and additional endogenous variables". The Stata Journal, 1,
    N. 1, pp. 1-13.

    Examples
    --------

    '''
    def __init__(self, y, x, w, yend, q, constant=True):

        # 1a. S2SLS --> \tilde{\delta}
        tsls = TSLS.BaseTSLS(y, x, yend, q=q, constant=constant)
        self.x = tsls.x
        self.z = tsls.z
        self.y = tsls.y
        self.yend = tsls.yend
        self.q = tsls.q
        self.h = tsls.h
        self.n, self.k = tsls.n, tsls.k
        #self.hth = tsls.hth

        w.A1 = get_A1_hom(w.sparse)

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

        # 2b. GM 2nd iteration --> \hat{\rho}
        moments = moments_hom(w, self.u)
        psi = get_vc_hom(w, self, lambda1, tsls_s.z)
        lambda2 = optim_moments(moments, psi)

        # Output
        self.betas = np.vstack((tsls_s.betas,lambda2))
        self.vm = get_omega_hom(w, lambda2, self, moments[0], psi, tsls_s.z)

def moments_hom(w, u):
    '''
    Compute G and g matrices for the spatial error model with homoscedasticity
    as in Drukker et al. [1]_ (p. 9).
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

    .. [1] Drukker, Prucha, I. R., Raciborski, R. (2010) "A command for
    estimating spatial-autoregressive models with spatial-autoregressive
    disturbances and additional endogenous variables". The Stata Journal, 1,
    N. 1, pp. 1-13.
    '''
    return _moments2eqs(w.A1, w.sparse, u)

def get_vc_hom(w, reg, lambdapar, z_s):
    '''
    VC matrix \psi of Spatial error with homoscedasticity. As in eq. (6) of
    Drukker et al. (2011) [2]_
    ...

    Parameters
    ----------
    w           :   W
                    Weights with A1 appended
    reg         :   reg
                    Regression object
    lambdapar   :   float
                    Spatial parameter estimated in previous step of the
                    procedure

    Returns
    -------

    psi         : array
                  2x2 VC matrix

    References
    ----------

    .. [1] Drukker, Prucha, I. R., Raciborski, R. (2010) "A command for
    estimating spatial-autoregressive models with spatial-autoregressive
    disturbances and additional endogenous variables". The Stata Journal, 1,
    N. 1, pp. 1-13.

    '''
    e = get_spFilter(w, lambdapar, reg.u)
    n = w.n*1.
    sig2 = np.dot(e.T, e) / n
    mu3 = np.sum(e**3) / n
    mu4 = np.sum(e**4) / n

    w.apat = w.A1 + w.A1.T
    w.wpwt = w.sparse + w.sparse.T
    prod = w.apat * w.apat
    tr11 = np.sum(prod.diagonal())
    prod = w.wpwt * w.apat
    tr12 = np.sum(prod.diagonal())
    prod = w.wpwt * w.wpwt
    tr22 = np.sum(prod.diagonal())
    a1, a2, p_s = _get_a1a2_filt(w, reg, lambdapar, w.apat, w.wpwt, e, z_s)
    prod = ['empty']
    vecd1 = np.array([w.A1.diagonal()]).T

    psi11 = (sig2**2 * tr11 / 2. + \
            sig2 * np.dot(a1.T, a1) + \
            (mu4 - 3 * sig2**2) * np.dot(vecd1.T, vecd1) + \
            mu3 * 2 * np.dot(a1.T, vecd1))
    psi22 = (sig2**2 * tr22 / 2. + \
            sig2 * np.dot(a2.T, a2)) # 3rd&4th terms=0 bc vecd2=0
    psi12 = (sig2**2 * tr12 / 2. + \
            sig2 * np.dot(a1.T, a2) + \
            mu3 * np.dot(a2.T, vecd1)) # 3rd term=0
    return np.array([[psi11[0][0], psi12[0][0]], [psi12[0][0], psi22[0][0]]]) / n

def get_omega_hom(w, lamb, reg_orig, G, psi, z_s):
    '''
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
    n = w.n*1.
    e = get_spFilter(w, lamb, reg_orig.u)
    sig2 = np.dot(e.T, e) / n
    mu3 = np.sum([i**3 for i in e]) / n
    #a1, a2, p_s = __get_a1a2(w, reg_orig, lamb)
    a1, a2, p_s = _get_a1a2_filt(w, reg_orig, lamb, w.apat, w.wpwt, e, z_s)
    j = np.dot(G, np.array([[1.], [2*lamb]]))
    q_hh = reg_orig.hth / n
    vecdA1 = np.reshape(w.A1.diagonal(), (w.n, 1))
    vecdW = np.zeros((w.n, 1))

    psiDD = sig2 * q_hh
    oDD = np.dot(psiDD, p_s)
    oDD = np.dot(p_s.T, oDD)

    psiRRi = la.inv(psi)
    oRR = np.dot(psiRRi, j)
    oRR = 1 / np.dot(j.T, oRR)

    psiDR = (sig2 * np.dot(reg_orig.h.T, np.hstack((a1, a2))) + \
            mu3 * np.dot(reg_orig.h.T, np.hstack((vecdA1, vecdW))) \
            ) / n
    oDR = np.dot(j, oRR)
    oDR = np.dot(psiRRi, oDR)
    oDR = np.dot(psiDR, oDR)
    oDR = np.dot(p_s.T, oDR)

    o_upper = np.hstack((oDD, oDR))
    o_lower = np.hstack((oDR.T, oRR))
    return np.vstack((o_upper, o_lower))

def _get_a1a2_filt(w, reg, lambdapar, apat, wpwt, e, z_s):
    '''
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

    import pysal
    db = pysal.open('examples/columbus.dbf','r')
    y = np.array([db.by_col('HOVAL')]).T
    x = np.array([db.by_col('INC')]).T
    w = pysal.open('examples/columbus.gal', 'r').read()
    w.transform='r' 
    w.A1 = get_A1_hom(w.sparse)
    q = []
    q.append(db.by_col("DISCBD"))
    q = np.array(q).T
    yd = []
    yd.append(db.by_col("CRIME"))
    yd = np.array(yd).T

    model = BaseGM_Endog_Error_Hom(y, x, w, yd, q) 
    print model.betas
    print model.vm

    """
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

