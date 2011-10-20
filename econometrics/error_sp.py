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

    >>> import pysal
    >>> import numpy as np
    >>> dbf = pysal.open('examples/columbus.dbf','r')
    >>> y = np.array([dbf.by_col('CRIME')]).T
    >>> x = np.array([dbf.by_col('INC')]).T
    >>> yend = np.array([dbf.by_col('HOVAL')]).T
    >>> q = np.array([dbf.by_col('DISCBD')]).T
    >>> w = pysal.open('examples/columbus.gal', 'r').read() 
    >>> w.transform='r'
    >>> model = GM_Endog_Error(y, x, yend, q, w, name_x=['inc'], name_y='crime', name_yend=['hoval'], name_q=['discbd'], name_ds='columbus')
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

    >>> reg = GM_Combo(y, X, w=w, name_y='crime', name_x=['income'], name_ds='columbus')

    Print the betas

    >>> print reg.name_z
    ['CONSTANT', 'income', 'W_crime', 'lambda']
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
        self.predy_sp, self.resid_sp = sp_att(w,self.y,\
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
                                            vm=vm, instruments=True)        

   

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
