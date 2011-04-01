"""
Spatial Error Models module
"""
from scipy.stats import norm
import numpy as np
from numpy import linalg as la
import pysal.spreg.ols as OLS
from pysal.spreg.diagnostics import se_betas
from pysal import lag_spatial
from utils import get_A1, optim_moments, get_spFilter, get_lags
import twosls as TSLS
import pysal.spreg.user_output as USER



class BaseGMSWLS:
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
    >>> model = BaseGMSWLS(y, x, w)
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
        w.A1 = get_A1(w.sparse)

        if constant:
            x = np.hstack((np.ones(y.shape),x))
        n, k = x.shape

        #1a. OLS --> \tilde{betas}
        ols = OLS.BaseOLS(y, x, constant=False)

        #1b. GMM --> \tilde{\lambda1}
        moments = _momentsGMSWLS(w, ols.u)
        lambda1 = optim_moments(moments)

        #2a. OLS -->\hat{betas}
        xs,ys = get_spFilter(w, lambda1, x),get_spFilter(w, lambda1, y)

        ols = OLS.BaseOLS(ys, xs, constant=False)

        #Output
        self.betas = np.vstack((ols.betas, np.array([[lambda1]])))
        self.sig2 = ols.sig2n
        self.u = ols.u

        self.se_betas, self.z, self.pvals = _inference(ols)
        self.step2OLS = ols

class GMSWLS(BaseGMSWLS):
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
    >>> model = GMSWLS(y, x, w, name_y='hoval', name_x=['income', 'crime'], name_ds='columbus')
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
        BaseGMSWLS.__init__(self, y, x, w, constant=constant) 
        self.title = "SPATIALLY WEIGHTED LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_x.append('lambda')


class BaseGSTSLS:
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
    >>> model = BaseGSTSLS(y, x, w, yend, q)
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

        if constant:
            x = np.hstack((np.ones(y.shape),x))
        n, k = x.shape

        #1a. TSLS --> \tilde{betas}
        tsls = TSLS.BaseTSLS(y, x, yend, q=q, constant=False)

        #1b. GMM --> \tilde{\lambda1}
        moments = _momentsGMSWLS(w, tsls.u)
        lambda1 = optim_moments(moments)

        #2a. OLS -->\hat{betas}
        x_s,y_s = get_spFilter(w, lambda1, x),get_spFilter(w, lambda1, y)
        yend_s = get_spFilter(w, lambda1, yend)

        tsls2 = TSLS.BaseTSLS(y_s, x_s, yend_s, h=tsls.h, constant=False)

        #Output
        self.betas = np.vstack((tsls2.betas, np.array([[lambda1]])))
        self.u = y - np.dot(tsls.z, tsls2.betas)
        sig2 = np.dot(tsls2.u.T,tsls2.u) / n
        self.vm = sig2 * la.inv(np.dot(tsls2.z.T,tsls2.z))
        self.se_betas = np.sqrt(self.vm.diagonal()).reshape(tsls2.betas.shape)
        zs = tsls2.betas / self.se_betas
        self.pvals = norm.sf(abs(zs)) * 2.


class GSTSLS(BaseGSTSLS):
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
    >>> model = GSTSLS(y, x, w, yend, q, name_x=['inc'], name_y='crime', name_yend=['hoval'], name_q=['discbd'], name_ds='columbus')
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
        BaseGSTSLS.__init__(self, y, x, w, yend, q, constant=constant)
        self.title = "GENERALIZED SPATIAL STAGE LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')
        self.name_q = USER.set_name_q(name_q, q)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        

class BaseGSTSLS_lag(BaseGSTSLS):
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

    >>> reg = BaseGSTSLS_lag(y, X, w)

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
    >>> reg = BaseGSTSLS_lag(y, X, w, yd, q)
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
            q = np.hstack((q, spatial_inst))
            yend = np.hstack((yend, yl))
        elif yend == None:                   # spatial instruments only
            q = get_lags(w, x, w_lags)
            yend = yl
        else:
            raise Exception, "invalid value passed to yend"
        BaseGSTSLS.__init__(self, y, x, w, yend, q, constant=constant)


class GSTSLS_lag(BaseGSTSLS_lag):
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

    >>> reg = GSTSLS_lag(y, X, w, name_y='crime', name_x=['income'], name_ds='columbus')

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
    >>> reg = GSTSLS_lag(y, X, w, yd, q, name_x=['inc'], name_y='crime', name_yend=['hoval'], name_q=['discbd'], name_ds='columbus')
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
        BaseGSTSLS_lag.__init__(self, y, x, w, yend, q, w_lags, constant)
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

def _momentsGMSWLS(w, u):

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
    dbf = pysal.open('examples/columbus.dbf','r')
    y = np.array([dbf.by_col('HOVAL')]).T
    x = np.array([dbf.by_col('INC'), dbf.by_col('CRIME')]).T
    w = pysal.open('examples/columbus.gal', 'r').read()
    w.transform='r' #Needed to match R

    print '\n\tGMSWLS model Example'
    model = GMSWLS(x, y, w)
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

