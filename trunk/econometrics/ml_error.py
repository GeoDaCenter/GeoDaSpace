"""
ML Estimation of Spatial Error Model
"""

__author__ = "Luc Anselin luc.anselin@asu.edu, Serge Rey srey@asu.edu"

import numpy as np
import numpy.linalg as la
import pysal as ps
from scipy.optimize import minimize_scalar
from pysal.spreg.utils import RegressionPropsY,RegressionPropsVM
import econometrics.diagnostics as DIAG  # uses latest, needs to switch to pysal
import econometrics.user_output as USER
import econometrics.summary_output as SUMMARY

__all__ = ["ML_Error"]

class BaseML_Error(RegressionPropsY,RegressionPropsVM):
    """
    ML estimation of the spatial error model (note no consistency 
    checks, diagnostics or constants added); Anselin (1988) [1]_
    
    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : Sparse matrix
                   Spatial weights sparse matrix 
    method       : string
                   if 'full', brute force calculation (full matrix expressions)
    epsilon      : float
                   tolerance criterion in mimimize_scalar function and inverse_product

    Attributes
    ----------
    betas        : array
                   kx1 array of estimated coefficients 
    lam          : float
                   estimate of spatial autoregressive coefficient
    u            : array
                   nx1 array of residuals
    e_filtered   : array
                   spatially filtered residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant, excluding the rho)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    method       : string
                   log Jacobian method
                   if 'full': brute force (full matrix computations)
    epsilon      : float
                   tolerance criterion used in minimize_scalar function and inverse_product
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (k+1 x k+1) - includes lambda
    vm1          : array
                   2x2 array of variance covariance for lambda, sigma
    sig2         : float
                   Sigma squared used in computations
    logll        : float
                   maximized log-likelihood (including constant terms)

    Examples
    ________
    
    >>> import numpy as np
    >>> import pysal as ps
    >>> db = ps.open(ps.examples.get_path("south.dbf"),'r')
    >>> y_name = "HR90"
    >>> y = np.array(db.by_col(y_name))
    >>> y.shape = (len(y),1)
    >>> x_names = ["RD90","PS90","UE90","DV90"]
    >>> x = np.array([db.by_col(var) for var in x_names]).T
    >>> x = np.hstack((np.ones((len(y),1)),x))
    >>> ww = ps.open(ps.examples.get_path("south_q.gal"))
    >>> w = ww.read()
    >>> ww.close()
    >>> w.transform = 'r'
    >>> mlerr = BaseML_Error(y,x,w)
    >>> mlerr.lam
    0.29907782498171109
    >>> mlerr.betas
    array([[ 6.14922483],
           [ 4.40242014],
           [ 1.77837126],
           [-0.37807312],
           [ 0.48578576],
           [ 0.29907782]])
    >>> mlerr.mean_y
    9.5492931620846928
    >>> mlerr.std_y
    7.0388508798387219
    >>> np.diag(mlerr.vm)
    array([ 1.06476526,  0.05548248,  0.04544514,  0.00614425,  0.01481356,
            0.00143001])
    >>> mlerr.sig2
    array([[ 32.40685441]])
    >>> mlerr.logll
    -4471.4070668878976


    References
    ----------

    .. [1] Anselin, L. (1988) "Spatial Econometrics: Methods and Models".
    Kluwer Academic Publishers. Dordrecht.

    """
    def __init__(self,y,x,w,method='full',epsilon=0.0000001):
        # set up main regression variables and spatial filters
        self.y = y
        self.x = x
        self.n, self.k = self.x.shape
        self.method = method
        self.epsilon = epsilon
        W = w.full()[0]
               
        ylag = ps.lag_spatial(w,self.y)
        xlag = ps.lag_spatial(w,self.x)
        
        # call minimizer using concentrated log-likelihood to get lambda
        
        if method.upper() == 'FULL':
            res = minimize_scalar(err_c_loglik,0.0,bounds=(-1.0,1.0),
                              args=(self.n,self.y,ylag,self.x,xlag,W),method='bounded',
                              tol=epsilon)
  
        self.lam = res.x
        
        # compute full log-likelihood, including constants
        ln2pi = np.log(2.0*np.pi)
        llik = -res.fun - self.n/2.0 * ln2pi - self.n/2.0 

        self.logll = llik
        
        # b, residuals and predicted values
        
        ys = self.y - self.lam*ylag
        xs = self.x - self.lam*xlag
        xsxs = np.dot(xs.T,xs)
        xsxsi = np.linalg.inv(xsxs)
        xsys = np.dot(xs.T,ys)
        b = np.dot(xsxsi,xsys)
        
        self.betas = np.vstack((b,self.lam))

        self.u = y - np.dot(x,b)
        self.predy = self.y - self.u

        # residual variance

        self.e_filtered = self.u - self.lam* ps.lag_spatial(w,self.u)
        self.sig2 = np.dot(self.e_filtered.T,self.e_filtered) / self.n
        
        # variance-covariance matrix betas
       
        varb = self.sig2 * xsxsi  

        # variance-covariance matrix lambda, sigma

        a = -self.lam * W
        np.fill_diagonal(a, 1.0)
        ai = la.inv(a)
        wai = np.dot(W,ai)
        tr1 = np.trace(wai)
        
        wai2 = np.dot(wai,wai)
        tr2 = np.trace(wai2)
        
        waiTwai = np.dot(wai.T,wai)
        tr3 = np.trace(waiTwai)
        
        v1 = np.vstack((tr2+tr3,
                       tr1/self.sig2))    
        v2 = np.vstack((tr1/self.sig2,
                       self.n/(2.0 * self.sig2**2)))
        
        v = np.hstack((v1,v2))
        
        self.vm1 = np.linalg.inv(v)
        
        # create variance matrix for beta, lambda
        vv = np.hstack((varb,np.zeros((self.k,1))))
        vv1 = np.hstack((np.zeros((1,self.k)),self.vm1[0,0]*np.ones((1,1))))
        
        self.vm = np.vstack((vv,vv1))
        
        self._cache = {}

class ML_Error(BaseML_Error):
    """
    ML estimation of the spatial lag model with all results and diagnostics; 
    Anselin (1988) [1]_
    
    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : Sparse matrix
                   Spatial weights sparse matrix 
    method       : string
                   if 'full', brute force calculation (full matrix expressions)
    epsilon      : float
                   tolerance criterion in mimimize_scalar function and inverse_product
    spat_diag    : boolean
                   if True, include spatial diagnostics
    vm           : boolean
                   if True, include variance-covariance matrix in summary
                   results
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output

    Attributes
    ----------
    betas        : array
                   (k+1)x1 array of estimated coefficients (rho first)
    lam          : float
                   estimate of spatial autoregressive coefficient
    u            : array
                   nx1 array of residuals
    e_filtered   : array
                   nx1 array of spatially filtered residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant, excluding lambda)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    method       : string
                   log Jacobian method
                   if 'full': brute force (full matrix computations)
    epsilon      : float
                   tolerance criterion used in minimize_scalar function and inverse_product
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    varb         : array
                   Variance covariance matrix (k+1 x k+1) - includes var(lambda)
    vm1          : array
                   variance covariance matrix for lambda, sigma (2 x 2)
    sig2         : float
                   Sigma squared used in computations
    logll        : float
                   maximized log-likelihood (including constant terms)
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    utu          : float
                   Sum of squared residuals
    std_err      : array
                   1xk array of standard errors of the betas    
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    title        : string
                   Name of the regression method used

    Examples
    ________
    
    >>> import numpy as np
    >>> import pysal as ps
    >>> db = ps.open(ps.examples.get_path("south.dbf"),'r')
    >>> ds_name = "south.dbf"
    >>> y_name = "HR90"
    >>> y = np.array(db.by_col(y_name))
    >>> y.shape = (len(y),1)
    >>> x_names = ["RD90","PS90","UE90","DV90"]
    >>> x = np.array([db.by_col(var) for var in x_names]).T
    >>> ww = ps.open(ps.examples.get_path("south_q.gal"))
    >>> w = ww.read()
    >>> ww.close()
    >>> w_name = "south_q.gal"
    >>> w.transform = 'r'    
    >>> mlerr = ML_Error(y,x,w,name_y=y_name,name_x=x_names,\
               name_w=w_name,name_ds=ds_name)
    >>> mlerr.betas
    array([[ 6.14922483],
           [ 4.40242014],
           [ 1.77837126],
           [-0.37807312],
           [ 0.48578576],
           [ 0.29907782]])
    >>> mlerr.lam
    0.29907782498171109
    >>> mlerr.mean_y
    9.5492931620846928
    >>> mlerr.std_y
    7.0388508798387219
    >>> np.diag(mlerr.vm)
    array([ 1.06476526,  0.05548248,  0.04544514,  0.00614425,  0.01481356,
            0.00143001])
    >>> mlerr.sig2
    array([[ 32.40685441]])
    >>> mlerr.logll
    -4471.4070668878976
    >>> mlerr.aic
    8952.8141337757952
    >>> mlerr.schwarz
    8979.0779458660618
    >>> mlerr.pr2
    0.3057664820479337
    >>> mlerr.utu
    48534.914822750536
    >>> mlerr.std_err
    array([ 1.03187463,  0.23554719,  0.21317867,  0.07838525,  0.12171098,
            0.03781546])
    >>> mlerr.z_stat
    [(5.9592751135203628, 2.5335925730458786e-09), (18.690182933636418, 5.9508613184155246e-78), (8.3421632972520374, 7.2943628086320888e-17), (-4.8232686335636661, 1.4122456267155579e-06), (3.9913060806046299, 6.5710406923833137e-05), (7.9088780562343812, 2.5971885919948263e-15)]
    >>> mlerr.name_y
    'HR90'
    >>> mlerr.name_x
    ['CONSTANT', 'RD90', 'PS90', 'UE90', 'DV90', 'lambda']
    >>> mlerr.name_w
    'south_q.gal'
    >>> mlerr.name_ds
    'south.dbf'
    >>> mlerr.title
    'MAXIMUM LIKELIHOOD SPATIAL ERROR (METHOD = FULL)'


    References
    ----------

    .. [1] Anselin, L. (1988) "Spatial Econometrics: Methods and Models".
    Kluwer Academic Publishers. Dordrecht.

    """
    def __init__(self,y,x,w,method='full',epsilon=0.0000001,\
                 spat_diag=False,vm=False,name_y=None,name_x=None,\
                 name_w=None,name_ds=None):
        n = USER.check_arrays(y,x)
        USER.check_y(y, n)
        USER.check_weights(w, y, w_required=True)        
        x_constant = USER.check_constant(x)
        BaseML_Error.__init__(self,y=y,x=x_constant,w=w,method=method,epsilon=epsilon)
        self.title = "MAXIMUM LIKELIHOOD SPATIAL ERROR" + " (METHOD = " + method.upper() + ")"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x)
        self.name_x.append('lambda')
        self.name_w = USER.set_name_w(name_w, w)
        self.aic = DIAG.akaike(reg=self)
        self.schwarz = DIAG.schwarz(reg=self)
        SUMMARY.ML_Error(reg=self,w=w,vm=vm,spat_diag=spat_diag)


def err_c_loglik(lam,n,y,ylag,x,xlag,W):
    #concentrated log-lik for error model, no constants, brute force
    ys = y - lam*ylag
    xs = x - lam*xlag
    ysys = np.dot(ys.T,ys)
    xsxs = np.dot(xs.T,xs)
    xsxsi = np.linalg.inv(xsxs)
    xsys = np.dot(xs.T,ys)
    x1 = np.dot(xsxsi,xsys)
    x2 = np.dot(xsys.T,x1)
    ee = ysys - x2
    sig2 = ee[0][0]/n
    nlsig2 = (n/2.0)*np.log(sig2)
    a = -lam * W
    np.fill_diagonal(a, 1.0)
    jacob = np.log(np.linalg.det(a))
    clik = nlsig2 - jacob  # this is the negative of the concentrated log lik for minimization
    return clik

def _test():
    import doctest
    start_suppress = np.get_printoptions()['suppress']
    np.set_printoptions(precision=8,suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)

if __name__ == "__main__":
    _test()
       
    import numpy as np
    import pysal as ps
    db = ps.open(ps.examples.get_path("south.dbf"),'r')
    ds_name = "south.dbf"
    y_name = "HR90"
    y = np.array(db.by_col(y_name))
    y.shape = (len(y),1)
    x_names = ["RD90","PS90","UE90","DV90"]
    x = np.array([db.by_col(var) for var in x_names]).T
    ww = ps.open(ps.examples.get_path("south_q.gal"))
    w = ww.read()
    ww.close()
    w_name = "south_q.gal"
    w.transform = 'r'
    mlerror = ML_Error(y,x,w,name_y=y_name,name_x=x_names,\
               name_w=w_name,name_ds=ds_name)
    print mlerror.summary