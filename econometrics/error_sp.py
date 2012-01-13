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
from utils import power_expansion, set_endog, iter_msg, sp_att
from utils import get_A1_hom, get_A2_hom, get_A1_het, optim_moments, get_spFilter, get_lags, _moments2eqs
from utils import RegressionPropsY
import twosls as TSLS
import user_output as USER



class BaseGM_Error(RegressionPropsY):
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
    array([[ 47.694635],
           [  0.710453],
           [ -0.550527],
           [  0.32573 ]])
    >>> np.around(model.se_betas, decimals=6)
    array([[ 12.412038],
           [  0.504443],
           [  0.178496]])
    >>> np.around(model.z, decimals=6)
    array([[ 3.842611],
           [ 1.408392],
           [-3.084247]])
    >>> np.around(model.pvals, decimals=6)
    array([[  1.22000000e-04],
           [  1.59015000e-01],
           [  2.04100000e-03]])
    >>> np.around(model.sig2, decimals=6)
    198.55957900000001

    """
    def __init__(self, y, x, w):

        #1a. OLS --> \tilde{betas}
        ols = OLS.BaseOLS(y=y, x=x)
        self.n, self.k = ols.x.shape
        self.x = ols.x
        self.y = ols.y

        #1b. GMM --> \tilde{\lambda1}
        moments = _momentsGM_Error(w, ols.u)
        lambda1 = optim_moments(moments)

        #2a. OLS -->\hat{betas}
        xs = get_spFilter(w, lambda1, self.x)
        ys = get_spFilter(w, lambda1, self.y)
        ols2 = OLS.BaseOLS(y=ys, x=xs, constant=False)

        #Output
        self.predy = np.dot(self.x, ols2.betas)
        self.u = y - self.predy
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

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import pysal
    >>> import numpy as np

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual OLS class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> dbf = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
    
    Extract the HOVAL column (home values) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array([dbf.by_col('HOVAL')]).T

    Extract CRIME (crime) and INC (income) vectors from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default pysal.spreg.OLS adds a vector of ones to the
    independent variables passed in, this can be overridden by passing
    constant=False.

    >>> names_to_extract = ['INC', 'CRIME']
    >>> x = np.array([dbf.by_col(name) for name in names_to_extract]).T

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will use
    ``columbus.gal``, which contains contiguity relationships between the
    observations in the Columbus dataset we are using throughout this example.
    Note that, in order to read the file, not only to open it, we need to
    append '.read()' at the end of the command.

    >>> w = pysal.open(pysal.examples.get_path("columbus.gal"), 'r').read() 
    
    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, his allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform='r'

    We are all set with the preliminars, we are good to run the model. In this
    case, we will need the variables and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> model = GM_Error(y, x, w, name_y='hoval', name_x=['income', 'crime'], name_ds='columbus')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas).

    >>> print model.name_x
    ['CONSTANT', 'income', 'crime', 'lambda']
    >>> np.around(model.betas, decimals=6)
    array([[ 47.694635],
           [  0.710453],
           [ -0.550527],
           [  0.32573 ]])
    >>> np.around(model.se_betas, decimals=6)
    array([[ 12.412038],
           [  0.504443],
           [  0.178496]])
    >>> np.around(model.z, decimals=6)
    array([[ 3.842611],
           [ 1.408392],
           [-3.084247]])
    >>> np.around(model.pvals, decimals=6)
    array([[  1.22000000e-04],
           [  1.59015000e-01],
           [  2.04100000e-03]])
    >>> np.around(model.sig2, decimals=6)
    198.55957900000001

    """
    def __init__(self, y, x, w,\
                 vm=False, name_y=None, name_x=None,\
                 name_w=None, name_ds=None):

        USER.check_arrays(y, x)
        USER.check_weights(w, y)
        USER.check_constant(x)
        BaseGM_Error.__init__(self, y=y, x=x, w=w) 
        self.title = "SPATIALLY WEIGHTED LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x)
        self.name_x.append('lambda')
        self.name_w = USER.set_name_w(name_w, w)
        self._get_diagnostics(w=w, beta_diag=True, vm=vm)

    def _get_diagnostics(self, beta_diag=True, w=None, vm=False):
        USER.DiagnosticBuilder.__init__(self, w=w, beta_diag=True,\
                                            nonspat_diag=False, lamb=True,\
                                            vm=vm, instruments=False)


class BaseGM_Endog_Error(RegressionPropsY):
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
    >>> model = BaseGM_Endog_Error(y, x, yend, q, w)
    >>> np.around(model.betas, decimals=5)
    array([[ 82.57297],
           [  0.58096],
           [ -1.44808],
           [  0.34992]])
    >>> np.around(model.se_betas, decimals=6)
    array([[ 16.138089],
           [  1.354476],
           [  0.786205]])

    '''
    def __init__(self, y, x, yend, q, w):

        #1a. TSLS --> \tilde{betas}
        tsls = TSLS.BaseTSLS(y=y, x=x, yend=yend, q=q)
        self.n, self.k = tsls.x.shape
        self.x = tsls.x
        self.y = tsls.y
        self.yend, self.z = tsls.yend, tsls.z

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
        self.predy = np.dot(tsls.z, tsls2.betas)
        self.u = y - self.predy
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

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import pysal
    >>> import numpy as np

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual OLS class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.  

    >>> dbf = pysal.open(pysal.examples.get_path("columbus.dbf"),'r')
    
    Extract the CRIME column (crime rates) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y = np.array([dbf.by_col('CRIME')]).T

    Extract INC (income) vector from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this model adds a vector of ones to the
    independent variables passed in, but this can be overridden by passing
    constant=False.

    >>> x = np.array([dbf.by_col('INC')]).T

    In this case we consider HOVAL (home value) is an endogenous regressor.
    We tell the model that this is so by passing it in a different parameter
    from the exogenous variables (x).

    >>> yend = np.array([dbf.by_col('HOVAL')]).T

    Because we have endogenous variables, to obtain a correct estimate of the
    model, we need to instrument for HOVAL. We use DISCBD (distance to the
    CBD) for this and hence put in the instruments parameter, 'q'.

    >>> q = np.array([dbf.by_col('DISCBD')]).T

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will use
    ``columbus.gal``, which contains contiguity relationships between the
    observations in the Columbus dataset we are using throughout this example.
    Note that, in order to read the file, not only to open it, we need to
    append '.read()' at the end of the command.

    >>> w = pysal.open(pysal.examples.get_path("columbus.gal"), 'r').read() 

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform='r'

    We are all set with the preliminars, we are good to run the model. In this
    case, we will need the variables and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> model = GM_Endog_Error(y, x, yend, q, w, name_x=['inc'], name_y='crime', name_yend=['hoval'], name_q=['discbd'], name_ds='columbus')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas). Also, this regression uses a two stage least
    squares estimation method that accounts for the endogeneity created by the
    endogenous variables included.

    >>> print model.name_z
    ['CONSTANT', 'inc', 'hoval', 'lambda']
    >>> np.around(model.betas, decimals=5)
    array([[ 82.57297],
           [  0.58096],
           [ -1.44808],
           [  0.34992]])
    >>> np.around(model.se_betas, decimals=6)
    array([[ 16.138089],
           [  1.354476],
           [  0.786205]])
    
    '''
    def __init__(self, y, x, yend, q, w,\
                 vm=False, name_y=None, name_x=None,\
                 name_yend=None, name_q=None,\
                 name_w=None, name_ds=None):

        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        USER.check_constant(x)
        BaseGM_Endog_Error.__init__(self, y=y, x=x, w=w, yend=yend, q=q)
        self.title = "GENERALIZED SPATIAL TWO STAGE LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')
        self.name_q = USER.set_name_q(name_q, q)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        self.name_w = USER.set_name_w(name_w, w)
        self._get_diagnostics(w=w, beta_diag=True, vm=vm)
     
    def _get_diagnostics(self, beta_diag=True, w=None, vm=False):
        USER.DiagnosticBuilder.__init__(self, w=w, beta_diag=True,\
                                            nonspat_diag=False, lamb=True,\
                                            vm=vm, instruments=True)        


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
    lag_q       : boolean
                  Optional. Whether to include or not as instruments spatial
                  lags of the additional instruments q. Set to True by default                  

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

    >>> reg = BaseGM_Combo(y, X, w=w)

    Print the betas

    >>> print np.around(np.hstack((reg.betas[:-1],np.sqrt(reg.vm.diagonal()).reshape(3,1))),3)
    [[ 39.059  11.86 ]
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
    >>> reg = BaseGM_Combo(y, X, yd, q, w)
    >>> betas = np.array([['CONSTANT'],['INC'],['HOVAL'],['W_CRIME']])
    >>> print np.hstack((betas, np.around(np.hstack((reg.betas[:-1], np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)))
    [['CONSTANT' '50.0944' '14.3593']
     ['INC' '-0.2552' '0.5667']
     ['HOVAL' '-0.6885' '0.3029']
     ['W_CRIME' '0.4375' '0.2314']]

        """
    def __init__(self, y, x, yend=None, q=None,\
                 w=None, w_lags=1, lag_q=True):

        yend2, q2 = set_endog(y, x, w, yend, q, w_lags, lag_q)
        BaseGM_Endog_Error.__init__(self, y=y, x=x, w=w, yend=yend2, q=q2)

class GM_Combo(BaseGM_Combo, USER.DiagnosticBuilder):
    """


    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import numpy as np
    >>> import pysal

    Open data on Columbus neighborhood crime (49 areas) using pysal.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    pysal.open() also reads data in CSV format; since the actual OLS class
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

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will use
    ``columbus.gal``, which contains contiguity relationships between the
    observations in the Columbus dataset we are using throughout this example.
    Note that, in order to read the file, not only to open it, we need to
    append '.read()' at the end of the command.

    >>> w = pysal.rook_from_shapefile("examples/columbus.shp")

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    Example only with spatial lag
    The Combo class runs an SARAR model, that is a spatial lag+error model.
    In this case we will run a simple version of that, where we have the
    spatial effects as well as exogenous variables. Since it is a spatial
    model, we to pass in the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> reg = GM_Combo(y, X, w=w, name_y='crime', name_x=['income'], name_ds='columbus')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas). Also, this regression uses a two stage least
    squares estimation method that accounts for the endogeneity created by the
    spatial lag of the dependent variable. We can check the betas:

    >>> print reg.name_z
    ['CONSTANT', 'income', 'W_crime', 'lambda']
    >>> print np.around(np.hstack((reg.betas[:-1],np.sqrt(reg.vm.diagonal()).reshape(3,1))),3)
    [[ 39.059  11.86 ]
     [ -1.404   0.391]
     [  0.467   0.2  ]]

    And lambda:

    >>> print 'lambda: ', np.around(reg.betas[-1], 3)
    lambda:  [-0.048]
        
    This class also allows the user to run a spatial lag+error model with the
    extra feature of including non-spatial endogenous regressors. This means
    that, in addition to the spatial lag and error, we consider some of the
    variables on the right-hand side of the equation as endogenous and we
    instrument for this. As an example, we will include HOVAL (home value) as
    endogenous and will instrument with DISCBD (distance to the CSB). We first
    need to read in the variables:

    >>> yd = []
    >>> yd.append(db.by_col("HOVAL"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T

    And then we can run and explore the model analogously to the previous combo:

    >>> reg = GM_Combo(y, X, yd, q, w, name_x=['inc'], name_y='crime', name_yend=['hoval'], name_q=['discbd'], name_ds='columbus')
    >>> print reg.name_z
    ['CONSTANT', 'inc', 'hoval', 'W_crime', 'lambda']
    >>> names = np.array(reg.name_z).reshape(5,1)
    >>> print np.hstack((names[0:4,:], np.around(np.hstack((reg.betas[:-1], np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)))
    [['CONSTANT' '50.0944' '14.3593']
     ['inc' '-0.2552' '0.5667']
     ['hoval' '-0.6885' '0.3029']
     ['W_crime' '0.4375' '0.2314']]

    >>> print 'lambda: ', np.around(reg.betas[-1], 3)
    lambda:  [ 0.254]
    """
    def __init__(self, y, x, yend=None, q=None,\
                 w=None, w_lags=1, lag_q=True,\
                 vm=False, name_y=None, name_x=None,\
                 name_yend=None, name_q=None,\
                 name_w=None, name_ds=None):

        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        USER.check_constant(x)
        BaseGM_Combo.__init__(self, y=y, x=x, w=w, yend=yend, q=q, w_lags=w_lags,\
                              lag_q=lag_q)
        self.predy_e, self.resid_sp = sp_att(w,self.y,\
                   self.predy,self.z[:,-1].reshape(self.n,1),self.betas[-2])        
        self.title = "GENERALIZED SPATIAL TWO STAGE LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_yend.append(USER.set_name_yend_sp(self.name_y))
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')
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
