"""
ML Estimation of Spatial Lag Model
"""

__author__ = "Luc Anselin luc.anselin@asu.edu, Serge Rey srey@asu.edu"

import numpy as np
import numpy.linalg as la
import pysal as ps
from scipy.optimize import minimize_scalar
from pysal.spreg.utils import RegressionPropsY,RegressionPropsVM,inverse_prod
import econometrics.diagnostics as DIAG  # uses latest, needs to switch to pysal
import econometrics.user_output as USER
import econometrics.summary_output as SUMMARY
from econometrics.w_utils import symmetrize

__all__ = ["ML_Lag"]

class BaseML_Lag(RegressionPropsY,RegressionPropsVM):
    """
    ML estimation of the spatial lag model (note no consistency 
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
                   (k+1)x1 array of estimated coefficients (rho first)
    rho          : float
                   estimate of spatial autoregressive coefficient
    u            : array
                   nx1 array of residuals
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
                   if 'ord' : Ord eigenvalue method
    epsilon      : float
                   tolerance criterion used in minimize_scalar function and inverse_product
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (k+1 x k+1)
    vm1          : array
                   Variance covariance matrix (k+2 x k+2) includes sigma2
    sig2         : float
                   Sigma squared used in computations
    logll        : float
                   maximized log-likelihood (including constant terms)
    predy_e      : array
                   predicted values from reduced form
    e_pred       : array
                   prediction errors using reduced form predicted values
                   
                   
    Examples
    ________
    
    >>> import numpy as np
    >>> import pysal as ps
    >>> db =  ps.open(ps.examples.get_path("baltim.dbf"),'r')
    >>> ds_name = "baltim.dbf"
    >>> y_name = "PRICE"
    >>> y = np.array(db.by_col(y_name)).T
    >>> y.shape = (len(y),1)
    >>> x_names = ["NROOM","NBATH","PATIO","FIREPL","AC","GAR","AGE","LOTSZ","SQFT"]
    >>> x = np.array([db.by_col(var) for var in x_names]).T
    >>> x = np.hstack((np.ones((len(y),1)),x))
    >>> ww = ps.open(ps.examples.get_path("baltim_q.gal"))
    >>> w = ww.read()
    >>> ww.close()
    >>> w.transform = 'r'
    >>> w_name = "baltim_q.gal"
    >>> mllag = BaseML_Lag(y,x,w,method='ord')
    >>> "{0:.6f}".format(mllag.rho)
    '0.425885'
    >>> mllag.betas
    array([[ 4.36748209],
           [ 0.75021751],
           [ 5.61164021],
           [ 7.04965543],
           [ 7.72458035],
           [ 6.12309367],
           [ 4.63746781],
           [-0.11073844],
           [ 0.06789793],
           [ 0.07935403],
           [ 0.42588451]])
    >>> "{0:.6f}".format(mllag.mean_y)
    '44.307180'
    >>> "{0:.6f}".format(mllag.std_y)
    '23.606077'
    >>> np.diag(mllag.vm1)
    array([  23.87164958,    1.12216517,    3.05930338,    7.34155519,
              5.66953642,    5.46976543,    2.86841645,    0.00258505,
              0.00021333,    0.02659621,    0.0032439 ,  220.12921049])
    >>> np.diag(mllag.vm)
    array([ 23.87164958,   1.12216517,   3.05930338,   7.34155519,
             5.66953642,   5.46976543,   2.86841645,   0.00258505,
             0.00021333,   0.02659621,   0.0032439 ])
    >>> "{0:.6f}".format(mllag.sig2)
    '151.458698'
    >>> "{0:.6f}".format(mllag.logll)
    '-832.937174'
    >>> mllag = BaseML_Lag(y,x,w)
    >>> "{0:.6f}".format(mllag.rho)
    '0.425885'
    >>> mllag.betas
    array([[ 4.36748209],
           [ 0.75021751],
           [ 5.61164021],
           [ 7.04965543],
           [ 7.72458035],
           [ 6.12309367],
           [ 4.63746781],
           [-0.11073844],
           [ 0.06789793],
           [ 0.07935403],
           [ 0.42588451]])
    >>> "{0:.6f}".format(mllag.mean_y)
    '44.307180'
    >>> "{0:.6f}".format(mllag.std_y)
    '23.606077'
    >>> np.diag(mllag.vm1)
    array([  23.87164958,    1.12216517,    3.05930338,    7.34155519,
              5.66953642,    5.46976543,    2.86841645,    0.00258505,
              0.00021333,    0.02659621,    0.0032439 ,  220.12921049])
    >>> np.diag(mllag.vm)
    array([ 23.87164958,   1.12216517,   3.05930338,   7.34155519,
             5.66953642,   5.46976543,   2.86841645,   0.00258505,
             0.00021333,   0.02659621,   0.0032439 ])
    >>> "{0:.6f}".format(mllag.sig2)
    '151.458698'
    >>> "{0:.6f}".format(mllag.logll)
    '-832.937174'
    
       
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
        ylag = ps.lag_spatial(w,y)
        # b0, b1, e0 and e1
        xtx = np.dot(self.x.T,self.x)
        xtxi = la.inv(xtx)
        xty = np.dot(self.x.T,self.y)
        xtyl = np.dot(self.x.T,ylag)
        b0 = np.dot(xtxi,xty)
        b1 = np.dot(xtxi,xtyl)
        e0 = self.y - np.dot(x,b0)
        e1 = ylag - np.dot(x,b1)
        methodML = method.upper()
        # call minimizer using concentrated log-likelihood to get rho
        if methodML in ['FULL','ORD']:
            if methodML == 'FULL':
                res = minimize_scalar(lag_c_loglik,0.0,bounds=(-1.0,1.0),
                              args=(self.n,e0,e1,W),method='bounded',
                              tol=epsilon)
            elif methodML == 'ORD':
                if w.asymmetry(intrinsic=False) == []:  # check on symmetry structure
                    ww = symmetrize(w)
                    WW = ww.todense()
                    evals = la.eigvalsh(WW)
                else:
                    evals = la.eigvals(W)
                res = minimize_scalar(lag_c_loglik_ord,0.0,bounds=(-1.0,1.0),
                              args=(self.n,e0,e1,evals),method='bounded',
                              tol=epsilon)
        else:
            print "{0} is an unsupported method".format(methodML)  # program will crash, need to catch
            self = None
            return

        self.rho = res.x[0][0]
    
        # compute full log-likelihood, including constants
        ln2pi = np.log(2.0*np.pi)
        llik = -res.fun - self.n/2.0 * ln2pi - self.n/2.0 
        self.logll = llik[0][0]
        
        # b, residuals and predicted values
        
        b = b0 - self.rho*b1
        self.betas = np.vstack((b,self.rho))   # rho added as last coefficient
        self.u = e0 - self.rho * e1
        self.predy = self.y - self.u
        
        xb = np.dot(x,b)
        
        self.predy_e = inverse_prod(w.sparse,xb,self.rho,inv_method="power_exp",threshold=epsilon)
        self.e_pred = self.y - self.predy_e
        
        # residual variance
        self._cache = {}
        self.sig2 = self.sig2n  #no allowance for division by n-k
        
        # information matrix
        a = -self.rho * W
        np.fill_diagonal(a, 1.0)
        ai = la.inv(a)
        wai = np.dot(W,ai)
        tr1 = np.trace(wai)

        wai2 = np.dot(wai,wai)
        tr2 = np.trace(wai2)
        
        waiTwai = np.dot(wai.T,wai)
        tr3 = np.trace(waiTwai)

        wpredy = ps.lag_spatial(w,self.predy_e)
        wpyTwpy = np.dot(wpredy.T,wpredy)
        xTwpy = np.dot(x.T,wpredy)
        
        # order of variables is beta, rho, sigma2
        
        v1 = np.vstack((xtx/self.sig2,xTwpy.T/self.sig2,np.zeros((1,self.k))))
        v2 = np.vstack((xTwpy/self.sig2,tr2+tr3+wpyTwpy/self.sig2,tr1/self.sig2))
        v3 = np.vstack((np.zeros((self.k,1)),tr1/self.sig2,self.n/(2.0 * self.sig2**2)))
        
        v = np.hstack((v1,v2,v3))
        
        self.vm1 = la.inv(v)  # vm1 includes variance for sigma2
        self.vm=self.vm1[:-1,:-1]  # vm is for coefficients only


class ML_Lag(BaseML_Lag):
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
                   if 'ord', Ord eigenvalue method
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
    rho          : float
                   estimate of spatial autoregressive coefficient
    u            : array
                   nx1 array of residuals
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
                   Variance covariance matrix (k+1 x k+1), all coefficients
    vm1          : array
                   Variance covariance matrix (k+2 x k+2), includes sig2
    sig2         : float
                   Sigma squared used in computations
    logll        : float
                   maximized log-likelihood (including constant terms)
    aic          : float
                   Akaike information criterion
    schwarz      : float
                   Schwarz criterion
    predy_e      : array
                   predicted values from reduced form
    e_pred       : array
                   prediction errors using reduced form predicted values
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    pr2_e        : float
                   Pseudo R squared (squared correlation between y and ypred_e
                   (using reduced form))
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
    >>> db =  ps.open(ps.examples.get_path("baltim.dbf"),'r')
    >>> ds_name = "baltim.dbf"
    >>> y_name = "PRICE"
    >>> y = np.array(db.by_col(y_name)).T
    >>> y.shape = (len(y),1)
    >>> x_names = ["NROOM","NBATH","PATIO","FIREPL","AC","GAR","AGE","LOTSZ","SQFT"]
    >>> x = np.array([db.by_col(var) for var in x_names]).T
    >>> ww = ps.open(ps.examples.get_path("baltim_q.gal"))
    >>> w = ww.read()
    >>> ww.close()
    >>> w_name = "baltim_q.gal"
    >>> w.transform = 'r'    
    >>> mllag = ML_Lag(y,x,w,name_y=y_name,name_x=x_names,\
               name_w=w_name,name_ds=ds_name)
    >>> mllag.betas
    array([[ 4.36748208],
           [ 0.75021751],
           [ 5.61164021],
           [ 7.04965543],
           [ 7.72458035],
           [ 6.12309367],
           [ 4.63746781],
           [-0.11073844],
           [ 0.06789793],
           [ 0.07935403],
           [ 0.42588451]])
    >>> "{0:.6f}".format(mllag.rho)
    '0.425885'
    >>> "{0:.6f}".format(mllag.mean_y)
    '44.307180'
    >>> "{0:.6f}".format(mllag.std_y)
    '23.606077'
    >>> np.diag(mllag.vm1)
    array([  23.87164958,    1.12216517,    3.05930338,    7.34155519,
              5.66953642,    5.46976543,    2.86841645,    0.00258505,
              0.00021333,    0.02659621,    0.0032439 ,  220.12921049])
    >>> np.diag(mllag.vm)
    array([ 23.87164958,   1.12216517,   3.05930338,   7.34155519,
             5.66953642,   5.46976543,   2.86841645,   0.00258505,
             0.00021333,   0.02659621,   0.0032439 ])
    >>> "{0:.6f}".format(mllag.sig2)
    '151.458698'
    >>> "{0:.6f}".format(mllag.logll)
    '-832.937174'
    >>> "{0:.6f}".format(mllag.aic)
    '1687.874348'
    >>> "{0:.6f}".format(mllag.schwarz)
    '1724.744787'
    >>> "{0:.6f}".format(mllag.pr2)
    '0.727081'
    >>> "{0:.6f}".format(mllag.pr2_e)
    '0.706198'
    >>> "{0:.6f}".format(mllag.utu)
    '31957.785345'
    >>> mllag.std_err
    array([ 4.88586221,  1.05932298,  1.74908644,  2.70953044,  2.38107884,
            2.33875297,  1.69364   ,  0.05084342,  0.01460569,  0.16308345,
            0.05695527])
    >>> mllag.z_stat
    [(0.89390201615523512, 0.37137431896163586), (0.70820469757581495, 0.47881814938957667), (3.2083264024394027, 0.0013350988198983671), (2.6017996804830945, 0.0092736002685895099), (3.2441514459565561, 0.0011780109434751905), (2.6181019379550383, 0.0088420386668909643), (2.7381662070921924, 0.0061782842812455737), (-2.1780287227448394, 0.029403898423607633), (4.6487325619022561, 3.3398091232666756e-06), (0.48658540505105097, 0.62655216850545958), (7.4775260351363597, 7.573476741490055e-14)]
    >>> mllag.name_y
    'PRICE'
    >>> mllag.name_x
    ['CONSTANT', 'NROOM', 'NBATH', 'PATIO', 'FIREPL', 'AC', 'GAR', 'AGE', 'LOTSZ', 'SQFT', 'W_PRICE']
    >>> mllag.name_w
    'baltim_q.gal'
    >>> mllag.name_ds
    'baltim.dbf'
    >>> mllag.title
    'MAXIMUM LIKELIHOOD SPATIAL LAG (METHOD = FULL)'
    >>> mllag = ML_Lag(y,x,w,method='ord',name_y=y_name,name_x=x_names,\
               name_w=w_name,name_ds=ds_name)
    >>> mllag.betas
    array([[ 4.36748208],
           [ 0.75021751],
           [ 5.61164021],
           [ 7.04965543],
           [ 7.72458035],
           [ 6.12309367],
           [ 4.63746781],
           [-0.11073844],
           [ 0.06789793],
           [ 0.07935403],
           [ 0.42588451]])
    >>> "{0:.6f}".format(mllag.rho)
    '0.425885'
    >>> "{0:.6f}".format(mllag.mean_y)
    '44.307180'
    >>> "{0:.6f}".format(mllag.std_y)
    '23.606077'
    >>> np.diag(mllag.vm1)
    array([  23.87164958,    1.12216517,    3.05930338,    7.34155519,
              5.66953642,    5.46976543,    2.86841645,    0.00258505,
              0.00021333,    0.02659621,    0.0032439 ,  220.12921049])
    >>> np.diag(mllag.vm)
    array([ 23.87164958,   1.12216517,   3.05930338,   7.34155519,
             5.66953642,   5.46976543,   2.86841645,   0.00258505,
             0.00021333,   0.02659621,   0.0032439 ])
    >>> "{0:.6f}".format(mllag.sig2)
    '151.458698'
    >>> "{0:.6f}".format(mllag.logll)
    '-832.937174'
    >>> "{0:.6f}".format(mllag.aic)
    '1687.874348'
    >>> "{0:.6f}".format(mllag.schwarz)
    '1724.744787'
    >>> "{0:.6f}".format(mllag.pr2)
    '0.727081'
    >>> "{0:.6f}".format(mllag.pr2_e)
    '0.706198'
    >>> "{0:.6f}".format(mllag.utu)
    '31957.785345'
    >>> mllag.std_err
    array([ 4.88586221,  1.05932298,  1.74908644,  2.70953044,  2.38107884,
            2.33875297,  1.69364   ,  0.05084342,  0.01460569,  0.16308345,
            0.05695527])
    >>> mllag.z_stat
    [(0.89390201615523512, 0.37137431896163586), (0.70820469757581495, 0.47881814938957667), (3.2083264024394027, 0.0013350988198983671), (2.6017996804830945, 0.0092736002685895099), (3.2441514459565561, 0.0011780109434751905), (2.6181019379550383, 0.0088420386668909643), (2.7381662070921924, 0.0061782842812455737), (-2.1780287227448394, 0.029403898423607633), (4.6487325619022561, 3.3398091232666756e-06), (0.48658540505105097, 0.62655216850545958), (7.4775260351363597, 7.573476741490055e-14)]
    >>> mllag.name_y
    'PRICE'
    >>> mllag.name_x
    ['CONSTANT', 'NROOM', 'NBATH', 'PATIO', 'FIREPL', 'AC', 'GAR', 'AGE', 'LOTSZ', 'SQFT', 'W_PRICE']
    >>> mllag.name_w
    'baltim_q.gal'
    >>> mllag.name_ds
    'baltim.dbf'
    >>> mllag.title
    'MAXIMUM LIKELIHOOD SPATIAL LAG (METHOD = ORD)'
    
       
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
        method = method.upper()
        if method in ['FULL','ORD']:
            BaseML_Lag.__init__(self,y=y,x=x_constant,w=w,method=method,epsilon=epsilon)
            self.k += 1  # increase by 1 to have correct aic and sc, include rho in count
            self.title = "MAXIMUM LIKELIHOOD SPATIAL LAG" + " (METHOD = " + method + ")"
            self.name_ds = USER.set_name_ds(name_ds)
            self.name_y = USER.set_name_y(name_y)
            self.name_x = USER.set_name_x(name_x, x)
            name_ylag = USER.set_name_yend_sp(self.name_y)
            self.name_x.append(name_ylag)  #rho changed to last position
            self.name_w = USER.set_name_w(name_w, w)
            self.aic = DIAG.akaike(reg=self)
            self.schwarz = DIAG.schwarz(reg=self)
            SUMMARY.ML_Lag(reg=self,w=w,vm=vm,spat_diag=spat_diag)
        else:
            raise Exception,"{0} is an unsupported method".format(method)


def lag_c_loglik(rho,n,e0,e1,W):
    #concentrated log-lik for lag model, no constants, brute force
    er = e0 - rho*e1
    sig2 = np.dot(er.T,er)/n
    nlsig2 = (n/2.0)*np.log(sig2)
    a = -rho * W
    np.fill_diagonal(a, 1.0)
    jacob = np.log(np.linalg.det(a))
    clik = nlsig2 - jacob  # this is the negative of the concentrated log lik for minimization
    return clik

def lag_c_loglik_ord(rho,n,e0,e1,evals):
    #concentrated log-lik for lag model, no constants, Ord eigenvalue method
    er = e0 - rho*e1
    sig2 = np.dot(er.T,er)/n
    nlsig2 = (n/2.0)*np.log(sig2)
    revals = rho * evals
    jacob = np.log(1-revals).sum()
    if isinstance(jacob,complex):
        jacob = jacob.real
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
    """
    db = ps.open(ps.examples.get_path("NAT.dbf"),'r')
    ds_name = "NAT.DBF"
    y_name = "HR90"
    y = np.array(db.by_col(y_name))
    y.shape = (len(y),1)
    x_names = ["RD90","PS90","UE90","DV90","MA90"]
    x = np.array([db.by_col(var) for var in x_names]).T
    ww = ps.open(ps.examples.get_path("nat_queen.gal"))
    w = ww.read()
    ww.close()
    w_name = "nat_queen.gal"
    """
    db =  ps.open(ps.examples.get_path("baltim.dbf"),'r')
    ds_name = "baltim.dbf"
    y_name = "PRICE"
    y = np.array(db.by_col(y_name)).T
    y.shape = (len(y),1)
    x_names = ["NROOM","NBATH","PATIO","FIREPL","AC","GAR","AGE","LOTSZ","SQFT"]
    x = np.array([db.by_col(var) for var in x_names]).T
    ww = ps.open(ps.examples.get_path("baltim_q.gal"))
    w = ww.read()
    ww.close()
    w_name = "baltim_q.gal"
    
    w.transform = 'r'
    mllag = ML_Lag(y,x,w,method='full',name_y=y_name,name_x=x_names,\
               name_w=w_name,name_ds=ds_name)
    print mllag.summary
    mllag1 = ML_Lag(y,x,w,method='ord',name_y=y_name,name_x=x_names,\
               name_w=w_name,name_ds=ds_name)
    print mllag1.summary   
    