'''
Hom family of models based on: 

    Drukker, D. M., Egger, P., Prucha, I. R. (2010)
    "On Two-step Estimation of a Spatial Autoregressive Model with Autoregressive
    Disturbances and Endogenous Regressors". Working paper.
    
Following:

    Anselin, L. (2011) "GMM Estimation of Spatial Error Autocorrelation with
    and without Heteroskedasticity".

'''
from scipy.stats import norm
from scipy import sparse as SP
import numpy as np
import multiprocessing as mp
import copy
from numpy import linalg as la
import pysal.spreg.ols as OLS
from pysal.spreg.diagnostics import se_betas
from pysal import lag_spatial
from utils import power_expansion, set_endog, iter_msg, sp_att
from utils import get_A1_hom, get_A2_hom, get_A1_het, optim_moments, get_spFilter, get_lags, _moments2eqs
from utils import RegressionProps
import twosls as TSLS
import user_output as USER


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
                  array of independent variables (with constant)
    xtx         : array
                  X.T * X
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
     [  0.4129   0.1835]]
    >>> print np.around(reg.vm, 4)
    [[  1.51340700e+02  -5.29060000e+00  -1.85650000e+00  -2.40000000e-03]
     [ -5.29060000e+00   2.46700000e-01   5.14000000e-02   3.00000000e-04]
     [ -1.85650000e+00   5.14000000e-02   3.21000000e-02  -1.00000000e-04]
     [ -2.40000000e-03   3.00000000e-04  -1.00000000e-04   3.37000000e-02]]
    '''

    def __init__(self, y, x, w,\
                 max_iter=1, epsilon=0.00001, A1='het'):
        if A1 == 'hom':
            w.A1 = get_A1_hom(w.sparse)
        elif A1 == 'hom_sc':
            w.A1 = get_A1_hom(w.sparse, scalarKP=True)
        elif A1 == 'het':
            w.A1 = get_A1_het(w.sparse)

        w.A2 = get_A2_hom(w.sparse)

        # 1a. OLS --> \tilde{\delta}
        ols = OLS.BaseOLS(y=y, x=x)
        self.x, self.y, self.n, self.k, self.xtx = ols.x, ols.y, ols.n, ols.k, ols.xtx

        # 1b. GM --> \tilde{\rho}
        moments = moments_hom(w, ols.u)
        lambda1 = optim_moments(moments)
        lambda_old = lambda1

        iteration, eps = 0, 1
        while iteration<max_iter and eps>epsilon:
            # 2a. SWLS --> \hat{\delta}
            x_s = get_spFilter(w,lambda_old,self.x)
            y_s = get_spFilter(w,lambda_old,self.y)
            ols_s = OLS.BaseOLS(y=y_s, x=x_s, constant=False)
            self.predy = np.dot(self.x, ols_s.betas)
            self.u = self.y - self.predy

            # 2b. GM 2nd iteration --> \hat{\rho}
            moments = moments_hom(w, self.u)
            psi = get_vc_hom(w, self, lambda_old)[0]
            lambda2 = optim_moments(moments, psi)
            eps = abs(lambda2 - lambda_old)
            lambda_old = lambda2
            iteration+=1

        self.iter_stop = iter_msg(iteration,max_iter)

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
                  array of independent variables (with constant)
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
     [  0.4129   0.1835]]


    '''
    def __init__(self, y, x, w,\
                 max_iter=1, epsilon=0.00001, A1='het',\
                 vm=False, name_y=None, name_x=None,\
                 name_w=None, name_ds=None):

        USER.check_arrays(y, x)
        USER.check_weights(w, y)
        USER.check_constant(x)
        BaseGM_Error_Hom.__init__(self, y=y, x=x, w=w, A1=A1,\
                max_iter=max_iter, epsilon=epsilon)
        self.title = "GENERALIZED SPATIAL LEAST SQUARES (Hom)"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x)
        self.name_x.append('lambda')
        self.name_w = USER.set_name_w(name_w, w)
        self._get_diagnostics(w=w, beta_diag=True, vm=vm)

    def _get_diagnostics(self, beta_diag=True, w=None, vm=False):
        USER.DiagnosticBuilder.__init__(self, w=w, beta_diag=True,\
                                            nonspat_diag=False,\
                                            vm=vm, instruments=False)


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
                  array of independent variables (with constant)
    z           : array
                  nxk array of variables (combination of x and yend)
    h           : array
                  nxl array of instruments (combination of x and q)
    hth         : array
                  h.T * h
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
    >>> reg = BaseGM_Endog_Error_Hom(y, X, yd, q, w, A1='hom_sc')
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 55.3658  23.496 ]
     [  0.4643   0.7382]
     [ -0.669    0.3943]
     [  0.4321   0.1927]]

    
    '''
    def __init__(self, y, x, yend, q, w,\
                 max_iter=1, epsilon=0.00001, A1='het', constant=True):

        if A1 == 'hom':
            w.A1 = get_A1_hom(w.sparse)
        elif A1 == 'hom_sc':
            w.A1 = get_A1_hom(w.sparse, scalarKP=True)
        elif A1 == 'het':
            w.A1 = get_A1_het(w.sparse)

        w.A2 = get_A2_hom(w.sparse)

        # 1a. S2SLS --> \tilde{\delta}
        tsls = TSLS.BaseTSLS(y=y, x=x, yend=yend, q=q, constant=constant)
        self.x, self.z, self.h, self.y, self.hth = tsls.x, tsls.z, tsls.h, tsls.y, tsls.hth
        self.yend, self.q, self.n, self.k = tsls.yend, tsls.q, tsls.n, tsls.k

        # 1b. GM --> \tilde{\rho}
        moments = moments_hom(w, tsls.u)
        lambda1 = optim_moments(moments)
        lambda_old = lambda1

        iteration, eps = 0, 1
        while iteration<max_iter and eps>epsilon:
            # 2a. GS2SLS --> \hat{\delta}
            x_s = get_spFilter(w,lambda_old,self.x)
            y_s = get_spFilter(w,lambda_old,self.y)
            yend_s = get_spFilter(w, lambda_old, self.yend)
            tsls_s = TSLS.BaseTSLS(y=y_s, x=x_s, yend=yend_s, h=self.h, constant=False)
            self.predy = np.dot(self.z, tsls_s.betas)
            self.u = self.y - self.predy

            # 2b. GM 2nd iteration --> \hat{\rho}
            moments = moments_hom(w, self.u)
            psi = get_vc_hom(w, self, lambda_old, tsls_s.z)[0]
            lambda2 = optim_moments(moments, psi)
            eps = abs(lambda2 - lambda_old)
            lambda_old = lambda2
            iteration+=1

        self.iter_stop = iter_msg(iteration,max_iter)            

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
                  array of independent variables (with constant)
    z           : array
                  nxk array of variables (combination of x and yend)
    h           : array
                  nxl array of instruments (combination of x and q)
    hth         : array
                  h.T * h
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

    >>> reg = GM_Endog_Error_Hom(y, X, yd, q, w, A1='hom_sc', name_x=['inc'], name_y='hoval', name_yend=['crime'], name_q=['discbd'], name_ds='columbus')
    >>> print reg.name_z
    ['CONSTANT', 'inc', 'crime', 'lambda']
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 55.3658  23.496 ]
     [  0.4643   0.7382]
     [ -0.669    0.3943]
     [  0.4321   0.1927]]

    '''
    def __init__(self, y, x, yend, q, w,\
                 max_iter=1, epsilon=0.00001, A1='het',\
                 vm=False, name_y=None, name_x=None,\
                 name_yend=None, name_q=None,\
                 name_w=None, name_ds=None, constant=True):

        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        USER.check_constant(x)
        BaseGM_Endog_Error_Hom.__init__(self, y=y, x=x, w=w, yend=yend, q=q,\
                A1=A1, max_iter=max_iter, epsilon=epsilon, constant=constant)
        self.title = "GENERALIZED SPATIAL TWO STAGE LEAST SQUARES (Hom)"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')  #listing lambda last
        self.name_q = USER.set_name_q(name_q, q)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.name_w = USER.set_name_w(name_w, w)
        self._get_diagnostics(w=w, beta_diag=True, vm=vm)
        
    def _get_diagnostics(self, beta_diag=True, w=None, vm=False):
        USER.DiagnosticBuilder.__init__(self, w=w, beta_diag=True,\
                                            nonspat_diag=False, lamb=True,\
                                            vm=vm, instruments=True)        


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
    lag_q       : boolean
                  Optional. Whether to include or not as instruments spatial
                  lags of the additional instruments q. Set to True by default                  
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
                  array of independent variables (with constant)
    z           : array
                  nxk array of variables (combination of x and yend)
    h           : array
                  nxl array of instruments (combination of x and q)
    hth         : array
                  h.T * h
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

    >>> reg = BaseGM_Combo_Hom(y, X, w=w, A1='hom_sc')
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 10.1254  15.2871]
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
    >>> reg = BaseGM_Combo_Hom(y, X, yd, q, w, A1='hom_sc')
    >>> betas = np.array([['CONSTANT'],['inc'],['crime'],['W_hoval'],['lambda']])
    >>> print np.hstack((betas, np.around(np.hstack((reg.betas, np.sqrt(reg.vm.diagonal()).reshape(5,1))),5)))
    [['CONSTANT' '111.7705' '67.75191']
     ['inc' '-0.30974' '1.16656']
     ['crime' '-1.36043' '0.6841']
     ['W_hoval' '-0.52908' '0.84428']
     ['lambda' '0.60116' '0.18605']]

    '''
    def __init__(self, y, x, yend=None, q=None,\
                 w=None, w_lags=1, lag_q=True,\
                 max_iter=1, epsilon=0.00001, A1='het'):
    
        yend2, q2 = set_endog(y, x, w, yend, q, w_lags, lag_q)
        BaseGM_Endog_Error_Hom.__init__(self, y=y, x=x, w=w, yend=yend2, q=q2, A1=A1,\
                                        max_iter=max_iter, epsilon=epsilon)

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
    lag_q       : boolean
                  Optional. Whether to include or not as instruments spatial
                  lags of the additional instruments q. Set to True by default                  
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
                  array of independent variables (with constant)
    z           : array
                  nxk array of variables (combination of x and yend)
    h           : array
                  nxl array of instruments (combination of x and q)
    hth         : array
                  h.T * h
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
    predy_sp    : array
                  nx1 array of spatially weighted predicted values
                  predy_sp = (I - \rho W)^{-1}predy
    resid_sp    : array
                  nx1 array of residuals considering predy_sp as predicted values                  
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

    >>> reg = GM_Combo_Hom(y, X, w=w, A1='hom_sc', name_x=['inc'],\
            name_y='hoval', name_yend=['crime'], name_q=['discbd'],\
            name_ds='columbus')
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 10.1254  15.2871]
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
    >>> reg = GM_Combo_Hom(y, X, yd, q, w, A1='hom_sc', \
            name_ds='columbus')
    >>> betas = np.array([['CONSTANT'],['inc'],['crime'],['W_hoval'],['lambda']])
    >>> print np.hstack((betas, np.around(np.hstack((reg.betas, np.sqrt(reg.vm.diagonal()).reshape(5,1))),5)))
    [['CONSTANT' '111.7705' '67.75191']
     ['inc' '-0.30974' '1.16656']
     ['crime' '-1.36043' '0.6841']
     ['W_hoval' '-0.52908' '0.84428']
     ['lambda' '0.60116' '0.18605']]

    '''
    def __init__(self, y, x, yend=None, q=None,\
                 w=None, w_lags=1, lag_q=True,\
                 max_iter=1, epsilon=0.00001, A1='het',\
                 vm=False, name_y=None, name_x=None,\
                 name_yend=None, name_q=None,\
                 name_w=None, name_ds=None):
    
        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        USER.check_constant(x)
        BaseGM_Combo_Hom.__init__(self, y=y, x=x, w=w, yend=yend, q=q,\
                    w_lags=w_lags, A1=A1, lag_q=lag_q,\
                    max_iter=max_iter, epsilon=epsilon)
        self.predy_sp, self.resid_sp = sp_att(w,self.y,self.predy,\
                             self.z[:,-1].reshape(self.n,1),self.betas[-2])        
        self.title = "GENERALIZED SPATIAL TWO STAGE LEAST SQUARES (Hom)"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_yend.append(USER.set_name_yend_sp(self.name_y))
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')  #listing lambda last
        self.name_q = USER.set_name_q(name_q, q)
        self.name_q.extend(USER.set_name_q_sp(self.name_x, w_lags, self.name_q, lag_q))
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.name_w = USER.set_name_w(name_w, w)
        self._get_diagnostics(w=w, beta_diag=True, vm=vm)
     
    def _get_diagnostics(self, beta_diag=True, w=None, vm=False):
        USER.DiagnosticBuilder.__init__(self, w=w, beta_diag=True,\
                                            nonspat_diag=False, lamb=True,\
                                            vm=vm, instruments=True,\
                                            spatial_lag=True)        


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
    oLL = la.inv(np.dot(j.T, np.dot(psii, j))) / n
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
    [DEPRECATED]
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
 
def _test():
    import doctest
    doctest.testmod()
   

if __name__ == '__main__':

    _test()


