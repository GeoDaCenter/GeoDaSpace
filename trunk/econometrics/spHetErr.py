import numpy as np
import numpy.linalg as la
import ols as OLS
import user_output as USER
import utils as GMM
import twosls as TSLS
from utils import power_expansion
from scipy import sparse as SP
from pysal import lag_spatial

class BaseGM_Error_Het:
    """
    GMM method for a spatial error model with heteroskedasticity (note: no
    consistency checks)

    Based on Arraiz et al [1]_

    ...

    Parameters
    ----------

    x           : array
                  nxk array with independent variables aligned with y
    y           : array
                  nx1 array with dependent variables
    w           : W
                  PySAL weights instance aligned with y and with instances S
                  and A1 created
    cycles      : int
                  Optional. Number of iterations of steps 2a. and 2b. Set to 1
                  by default

    Attributes
    ----------

    x           : array
                  nxk array of independent variables
    y           : array
                  nx1 array of dependent variable  
    betas       : array
                  (k+1)x1 array with estimates for betas and lambda
    n           : int
                  Number of observations
    k           : int
                  Number of variables (constant included)
    u           : array
                  nx1 array with residuals
    vm          : array
                  (k+1)x(k+1) variance-covariance matrix
    predy       : array
                  nx1 array of predicted values

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

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
    >>> reg = BaseGM_Error_Het(y, X, w)
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 48.012   11.4405]
     [  0.7119   0.3653]
     [ -0.5597   0.1609]
     [  0.4259   0.2119]]
    """

    def __init__(self,y,x,w,cycles=1,constant=True): 
        
        #1a. OLS --> \tilde{betas}
        ols = OLS.BaseOLS(y, x, constant=constant)
        self.x = ols.x
        self.y = ols.y
        self.n, self.k = ols.n, ols.k

        w.A1 = GMM.get_A1_het(w.sparse)


        #1b. GMM --> \tilde{\lambda1}
        moments = moments_het(w, ols.u)
        lambda1 = GMM.optim_moments(moments)

        #1c. GMM --> \tilde{\lambda2}
        self.u = ols.u
        sigma = get_psi_sigma(w, ols.u, lambda1)
        vc1 = get_vc_het(w, sigma)
        lambda2 = GMM.optim_moments(moments,vc1)
        lambda2 = lambda1  # MIGHT need this to match Stata code
       
        #2a. reg -->\hat{betas}
        xs = GMM.get_spFilter(w, lambda2, self.x)
        ys = GMM.get_spFilter(w, lambda2, self.y)
        ols_s = OLS.BaseOLS(ys, xs, constant=False)
        self.predy = np.dot(self.x, ols_s.betas)
        self.u = self.y - self.predy

        #2b. GMM --> \hat{\lambda}
        sigma = get_psi_sigma(w, ols_s.u, lambda2)
        vc2 = get_vc_het(w, sigma)
        moments_i = moments_het(w, self.u)
        lambda3 = GMM.optim_moments(moments_i, vc2)

        sigma = get_psi_sigma(w, ols_s.u, lambda3)
        vc3 = get_vc_het(w, sigma)
        G = moments_i[0]

        self.vm = get_vm_het(G, lambda3, self, w, vc3)
        self.betas = np.vstack((ols_s.betas, lambda3))
        self._cache = {}

        """
        #The following code will give results that match Stata

        ones = np.ones(y.shape)
        reg = BaseGM_Endog_Error_Het(y, x=ones, w=w, yend=x, q=x,
                cycles=cycles, constant=False)
        self.x = reg.z
        self.y = reg.y
        self.n, self.k = reg.n, reg.k
        self.betas = reg.betas
        self.vm = reg.vm
        self.u = reg.u
        self.predy = reg.predy
        self._cache = {}
        """



class GM_Error_Het(BaseGM_Error_Het):
    """
    GMM method for a spatial error model with heteroskedasticity
    Based on Arraiz et al [1]_

    ...

    Parameters
    ----------

    x           : array
                  nxk array with independent variables aligned with y
    y           : array
                  nx1 array with dependent variables
    w           : W
                  PySAL weights instance aligned with y and with instances S
                  and A1 created
    cycles      : int
                  Optional. Number of iterations of steps 2a. and 2b. Set to 1
                  by default
    name_ds     : string
                  dataset's name
    name_y      : string
                  Dependent variable's name
    name_x      : tuple
                  Independent variables' names

    Attributes
    ----------

    x           : array
                  nxk array of independent variables
    y           : array
                  nx1 array of dependent variable  
    betas       : array
                  (k+1)x1 array with estimates for betas and lambda
    n           : int
                  Number of observations
    k           : int
                  Number of variables (constant included)
    u           : array
                  nx1 array with residuals
    vm          : array
                  (k+1)x(k+1) variance-covariance matrix
    predy       : array
                  nx1 array of predicted values

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

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
    >>> reg = GM_Error_Het(y, X, w, name_y='home value', name_x=['income', 'crime'], name_ds='columbus')
    >>> print reg.name_x
    ['CONSTANT', 'income', 'crime', 'lambda']
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 48.012   11.4405]
     [  0.7119   0.3653]
     [ -0.5597   0.1609]
     [  0.4259   0.2119]]

    """

    def __init__(self, y, x, w, cycles=1, constant=True,\
                        name_y=None, name_x=None, name_ds=None):
        USER.check_arrays(y, x)
        USER.check_weights(w, y)
        BaseGM_Error_Het.__init__(self, y, x, w, cycles=cycles, constant=constant)
        self.title = "SPATIALLY WEIGHTED LEAST SQUARES"        
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_x.append('lambda')
        

class BaseGM_Endog_Error_Het:
    """
    GMM method for a spatial error model with heteroskedasticity and
    endogenous variables (note: no consistency checks)

    Based on Arraiz et al [1]_

    ...

    Parameters
    ----------

    x           : array
                  nxk array with independent variables aligned with y
    y           : array
                  nx1 array with dependent variables
    yend        : array
                  Endogenous variables
    q           : array
                  array of instruments for yend (note: this should not contain
                  any variables from x;
    w           : W
                  PySAL weights instance aligned with y
    cycles      : int
                  Optional. Number of iterations of steps 2a. and 2b. Set to 1
                  by default

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

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

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
    >>> reg = BaseGM_Endog_Error_Het(y, X, w, yd, q)
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 55.4616  28.9106]
     [  0.468    0.7727]
     [ -0.673    0.4685]
     [  0.4126   0.1776]]
    """

    def __init__(self,y,x,w,yend,q,cycles=1,constant=True): 
        #1a. reg --> \tilde{betas} 
        tsls = TSLS.BaseTSLS(y, x, yend, q=q, constant=constant)
        self.x = tsls.x
        self.z = tsls.z
        self.h = tsls.h
        self.y = tsls.y
        self.yend = tsls.yend
        self.q = tsls.q
        self.n, self.k = tsls.n, tsls.k

        w.A1 = GMM.get_A1_het(w.sparse)

        #1b. GMM --> \tilde{\lambda1}
        moments = moments_het(w, tsls.u)
        lambda1 = GMM.optim_moments(moments)

        #1c. GMM --> \tilde{\lambda2}
        self.u = tsls.u
        vc1 = get_vc_het_tsls(w, self, lambda1, tsls.pfora1a2, filt=False)
        lambda2 = GMM.optim_moments(moments,vc1)
        lambda2 = lambda1  # need this to match Stata code
       
        #2a. reg -->\hat{betas}
        xs = GMM.get_spFilter(w, lambda2, self.x)
        ys = GMM.get_spFilter(w, lambda2, self.y)
        yend_s = GMM.get_spFilter(w, lambda2, self.yend)
        tsls_s = TSLS.BaseTSLS(ys, xs, yend_s, h=self.h, constant=False)
        self.predy = np.dot(self.z, tsls_s.betas)
        self.u = self.y - self.predy

        #2b. GMM --> \hat{\lambda}
        vc2 = get_vc_het_tsls(w, self, lambda2, tsls_s.pfora1a2, filt=True)
        moments_i = moments_het(w, self.u)
        lambda3 = GMM.optim_moments(moments_i, vc2)

        xs = GMM.get_spFilter(w, lambda3, self.x)
        yend_s = GMM.get_spFilter(w, lambda3, self.yend)
        P = get_P_hat(self.h, tsls.hthi, np.hstack((xs, yend_s)), self.n)
        vc3 = get_vc_het_tsls(w, self, lambda3, P, filt=True)
        G = moments_i[0]

        self.vm = get_Omega_GS2SLS(w, lambda3, self, G, vc3, P, filt=True)
        self.betas = np.vstack((tsls_s.betas, lambda3))
        self._cache = {}


class GM_Endog_Error_Het(BaseGM_Endog_Error_Het):
    """
    GMM method for a spatial error model with heteroskedasticity and endogenous variables

    Based on Arraiz et al [1]_

    ...

    Parameters
    ----------

    x           : array
                  nxk array with independent variables aligned with y
    y           : array
                  nx1 array with dependent variables
    yend        : array
                  Endogenous variables
    q           : array
                  array of instruments for yend (note: this should not contain
                  any variables from x;
    w           : W
                  PySAL weights instance aligned with y
    cycles      : int
                  Optional. Number of iterations of steps 2a. and 2b. Set to 1
                  by default
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

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.


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
    >>> reg = GM_Endog_Error_Het(y, X, w, yd, q, name_x=['inc'], name_y='hoval', name_yend=['crime'], name_q=['discbd'], name_ds='columbus')
    >>> print reg.name_z
    ['CONSTANT', 'inc', 'crime', 'lambda']
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 55.4616  28.9106]
     [  0.468    0.7727]
     [ -0.673    0.4685]
     [  0.4126   0.1776]]

    """
    def __init__(self, y, x, w, yend, q, cycles=1, constant=True,\
                        name_y=None, name_x=None, name_yend=None,\
                        name_q=None, name_ds=None):
        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        BaseGM_Endog_Error_Het.__init__(self, y, x, w, yend, q, cycles=cycles,\
                                       constant=constant)
        self.title = "GENERALIZED SPATIAL TWO STAGE LEAST SQUARES"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, x, constant)
        self.name_yend = USER.set_name_yend(name_yend, yend)
        self.name_z = self.name_x + self.name_yend
        self.name_z.append('lambda')  #listing lambda last
        self.name_q = USER.set_name_q(name_q, q)
        self.name_h = USER.set_name_h(self.name_x, self.name_q)
        

class BaseGM_Combo_Het(BaseGM_Endog_Error_Het):
    """
    GMM method for a spatial lang and error model with heteroskedasticity and
    endogenous variables  (note: no consistency checks) 

    Based on Arraiz et al [1]_

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
    cycles      : int
                  Optional. Number of iterations of steps 2a. and 2b. Set to 1
                  by default

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

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

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

    >>> reg = BaseGM_Combo_Het(y, X, w)
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 10.0489  14.1696]
     [  1.5714   0.3742]
     [  0.1524   0.3982]
     [  0.2126   0.3932]]

    Example with both spatial lag and other endogenous variables

    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> reg = BaseGM_Combo_Het(y, X, w, yd, q)
    >>> betas = np.array([['CONSTANT'],['inc'],['crime'],['lag_hoval'],['lambda']])
    >>> print np.hstack((betas, np.around(np.hstack((reg.betas, np.sqrt(reg.vm.diagonal()).reshape(5,1))),5)))
    [['CONSTANT' '110.1897' '63.89236']
     ['inc' '-0.28381' '1.16689']
     ['crime' '-1.3613' '0.72178']
     ['lag_hoval' '-0.49522' '0.7502']
     ['lambda' '0.65217' '0.15309']]
    """

    def __init__(self, y, x, w, yend=None, q=None, w_lags=1,\
                    constant=True, cycles=1):
        # Create spatial lag of y
        yl = lag_spatial(w, y)
        if issubclass(type(yend), np.ndarray):  # spatial and non-spatial instruments
            lag_vars = np.hstack((x, q))
            spatial_inst = GMM.get_lags(w ,lag_vars, w_lags)
            q = np.hstack((q, spatial_inst))
            yend = np.hstack((yend, yl))
        elif yend == None:                   # spatial instruments only
            q = GMM.get_lags(w, x, w_lags)
            yend = yl
        else:
            raise Exception, "invalid value passed to yend"
        BaseGM_Endog_Error_Het.__init__(self, y, x, w, yend, q, cycles=cycles, constant=constant)

class GM_Combo_Het(BaseGM_Combo_Het):
    """
    GMM method for a spatial lang and error model with heteroskedasticity and
    endogenous variables  (note: no consistency checks) 

    Based on Arraiz et al [1]_

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
    cycles      : int
                  Optional. Number of iterations of steps 2a. and 2b. Set to 1
                  by default
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

    >>> reg = GM_Combo_Het(y, X, w, name_y='hoval', name_x=['income'], name_ds='columbus')
    >>> print reg.name_z
    ['CONSTANT', 'income', 'lag_hoval', 'lambda']
    >>> print np.around(np.hstack((reg.betas,np.sqrt(reg.vm.diagonal()).reshape(4,1))),4)
    [[ 10.0489  14.1696]
     [  1.5714   0.3742]
     [  0.1524   0.3982]
     [  0.2126   0.3932]]
        
    Example with both spatial lag and other endogenous variables

    >>> yd = []
    >>> yd.append(db.by_col("CRIME"))
    >>> yd = np.array(yd).T
    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T
    >>> reg = GM_Combo_Het(y, X, w, yd, q, name_x=['inc'], name_y='hoval', name_yend=['crime'], name_q=['discbd'], name_ds='columbus')
    >>> print reg.name_z
    ['CONSTANT', 'inc', 'crime', 'lag_hoval', 'lambda']
    >>> print np.round(reg.betas,4)
    [[ 110.1897]
     [  -0.2838]
     [  -1.3613]
     [  -0.4952]
     [   0.6522]]
    
    """
    
    def __init__(self, y, x, w, yend=None, q=None, w_lags=1,\
                    constant=True, cycles=1,\
                    name_y=None, name_x=None, name_yend=None,\
                    name_q=None, name_ds=None):

        USER.check_arrays(y, x, yend, q)
        USER.check_weights(w, y)
        BaseGM_Combo_Het.__init__(self, y, x, w, yend, q, w_lags,\
                                           constant, cycles)
        self.title = "GENERALIZED SPATIAL TWO STAGE LEAST SQUARES"        
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


def moments_het(w, u):
    """
    Function to compute all six components of the system of equations for a
    spatial error model with heteroskedasticity estimated by GMM as in Arraiz
    et al [1]_

    Scipy sparse matrix version. It implements eqs. A.1 in Appendix A of
    Arraiz et al. (2007) by using all matrix manipulation
    
    [g1] + [G11 G12] *  [\lambda]    = [0]
    [g2]   [G21 G22]    [\lambda^2]    [0]

    NOTE: 'residuals' has been renamed 'u' to fit paper notation

    ...
    
    Parameters
    ----------

    w           : W
                  Spatial weights instance (requires 'S' and 'A1')

    u           : array
                  Residuals. nx1 array assumed to be aligned with w
 
    Attributes
    ----------

    moments     : list
                  List of two arrays corresponding to the matrices 'G' and
                  'g', respectively.

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

    """
    return GMM._moments2eqs(w.A1, w.sparse, u)

def get_psi_sigma(w, u, lamb):
    """
    Computes the Sigma matrix needed to compute Psi

    Parameters
    ----------
    w           : W
                  Spatial weights instance (requires 'S' and 'A1')

    u           : array
                  nx1 vector of residuals

    lamb        : float
                  Lambda

    """

    e = (u - lamb * (w.sparse * u)) ** 2
    E = SP.dia_matrix((e.flat,0), shape=(w.n,w.n))
    return E.tocsr()

def get_vc_het(w, E):
    """
    Computes the VC matrix Psi based on lambda as in Arraiz et al [1]_:

    ..math::

        \tilde{Psi} = \left(\begin{array}{c c}
                            \psi_{11} & \psi_{12} \\
                            \psi_{21} & \psi_{22} \\
                      \end{array} \right)

    NOTE: psi12=psi21

    ...

    Parameters
    ----------

    w           : W
                  Spatial weights instance (requires 'S' and 'A1')

    E           : sparse matrix
                  Sigma
 
    Returns
    -------

    Psi         : array
                  2x2 array with estimator of the variance-covariance matrix

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

    """
    aPatE = (w.A1 + w.A1.T) * E
    wPwtE = (w.sparse + w.sparse.T) * E

    psi11 = aPatE * aPatE
    psi12 = aPatE * wPwtE
    psi22 = wPwtE * wPwtE 
    psi = map(np.sum, [psi11.diagonal(), psi12.diagonal(), psi22.diagonal()])
    return np.array([[psi[0], psi[1]], [psi[1], psi[2]]]) / (2. * w.n)

def get_vm_het(G, lamb, reg, w, psi):
    """
    Computes the variance-covariance matrix Omega as in Arraiz et al [1]_:
    ...

    Parameters
    ----------

    G           : array
                  G from moments equations

    lamb        : float
                  Final lambda from spHetErr estimation

    reg         : regression object
                  output instance from a regression model

    u           : array
                  nx1 vector of residuals

    w           : W
                  Spatial weights instance

    psi         : array
                  2x2 array with the variance-covariance matrix of the moment equations
 
    Returns
    -------

    vm          : array
                  (k+1)x(k+1) array with the variance-covariance matrix of the parameters

    References
    ----------

    .. [1] Arraiz, I., Drukker, D. M., Kelejian, H., Prucha, I. R. (2010) "A
    Spatial Cliff-Ord-Type Model with Heteroskedastic Innovations: Small and
    Large Sample Results". Journal of Regional Science, Vol. 60, No. 2, pp.
    592-614.

    """

    J = np.dot(G, np.array([[1],[2 * lamb]]))
    Zs = GMM.get_spFilter(w,lamb,reg.x)
    ZstEZs = np.dot((Zs.T * get_psi_sigma(w, reg.u, lamb)), Zs)
    ZsZsi = la.inv(np.dot(Zs.T,Zs))
    omega11 = w.n * np.dot(np.dot(ZsZsi,ZstEZs),ZsZsi)
    omega22 = la.inv(np.dot(np.dot(J.T,la.inv(psi)),J))
    zero = np.zeros((reg.k,1),float)
    vm = np.vstack((np.hstack((omega11, zero)),np.hstack((zero.T, omega22)))) / w.n
    return vm

def get_P_hat(h, hthi, zf, n):
    """
    P_hat from Appendix B, used for a1 a2, using filtered Z
    """
    htzf = np.dot(h.T, zf)
    P1 = np.dot(hthi, htzf)
    P2 = np.dot(htzf.T, P1)
    P2i = la.inv(P2)
    return n*np.dot(P1, P2i)

def get_a1a2(w, reg, lambdapar, P, filt):
    """
    Computes the a1 in psi assuming residuals come from original regression
    ...

    Parameters
    ----------

    w           : W
                  Spatial weights instance 

    reg         : TSLS
                  Two stage least quare regression instance
                  
    lambdapar   : float
                  Spatial autoregressive parameter
 
    Returns
    -------

    [a1, a2]    : list
                  a1 and a2 are two nx1 array in psi equation

    References
    ----------

    .. [1] Anselin, L. GMM Estimation of Spatial Error Autocorrelation with Heteroskedasticity
    
    """        
    zst = GMM.get_spFilter(w, lambdapar, reg.z).T
    us = GMM.get_spFilter(w, lambdapar, reg.u)
    alpha1 = (-2.0/w.n) * (np.dot((zst * w.A1), us))
    alpha2 = (-1.0/w.n) * (np.dot((zst * (w.sparse + w.sparse.T)), us))
    a1t = np.dot(np.dot(reg.h, P), alpha1).T
    a2t = np.dot(np.dot(reg.h, P), alpha2).T
    if not filt:
        a1t = power_expansion(w, a1t.T, lambdapar, transpose=True)
        a2t = power_expansion(w, a2t.T, lambdapar, transpose=True)
    return [a1t.T, a2t.T]

def get_vc_het_tsls(w, reg, lambdapar, P, filt):

    sigma = get_psi_sigma(w, reg.u, lambdapar)
    vc1 = get_vc_het(w, sigma)
    a1, a2 = get_a1a2(w, reg, lambdapar, P, filt=filt)
    a1s = a1.T * sigma
    a2s = a2.T * sigma
    psi11 = float(np.dot(a1s, a1))
    psi12 = float(np.dot(a1s, a2))
    psi21 = float(np.dot(a2s, a1))
    psi22 = float(np.dot(a2s, a2))
    psi = np.array([[psi11, psi12], [psi21, psi22]]) / w.n
    return vc1 + psi

def get_Omega_GS2SLS(w, lamb, reg, G, psi, P, filt):
    """
    Computes the variance-covariance matrix for GS2SLS:
    ...

    Parameters
    ----------

    w           : W
                  Spatial weights instance 

    lamb        : float
                  Spatial autoregressive parameter
                  
    reg         : GSTSLS
                  Generalized Spatial two stage least quare regression instance
    G           : array
                  Moments
    psi         : array
                  Weighting matrix
 
    Returns
    -------

    omega       : array
                  (k+1)x(k+1)                 
    """
    sigma=get_psi_sigma(w, reg.u, lamb)
    psi_dd_1=(1.0/w.n) * reg.h.T * sigma 
    psi_dd = np.dot(psi_dd_1, reg.h)
    a1a2=get_a1a2(w, reg, lamb, P, filt=filt)
    psi_dl=np.dot(psi_dd_1,np.hstack(tuple(a1a2)))
    psi_o=np.hstack((np.vstack((psi_dd, psi_dl.T)), np.vstack((psi_dl, psi))))
    psii=la.inv(psi)
   
    j = np.dot(G, np.array([[1.], [2*lamb]]))
    jtpsii=np.dot(j.T, psii)
    jtpsiij=np.dot(jtpsii, j)
    jtpsiiji=la.inv(jtpsiij)
    omega_1=np.dot(jtpsiiji, jtpsii)
    omega_2=np.dot(np.dot(psii, j), jtpsiiji)
    om_1_s=omega_1.shape
    om_2_s=omega_2.shape
    p_s=P.shape
    
    omega_left=np.hstack((np.vstack((P.T, np.zeros((om_1_s[0],p_s[0])))), 
               np.vstack((np.zeros((p_s[1], om_1_s[1])), omega_1))))
    omega_right=np.hstack((np.vstack((P, np.zeros((om_2_s[0],p_s[1])))), 
               np.vstack((np.zeros((p_s[0], om_2_s[1])), omega_2))))
    omega=np.dot(np.dot(omega_left, psi_o), omega_right)    
    return omega / w.n
                    

def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
    import numpy as np
    import pysal
    db=pysal.open("examples/columbus.dbf","r")
    y = np.array(db.by_col("CRIME"))
    y = np.reshape(y, (49,1))
    X = []
    X.append(db.by_col("INC"))
    X = np.array(X).T
    yd = []
    yd.append(db.by_col("HOVAL"))
    yd = np.array(yd).T
    q = []
    q.append(db.by_col("DISCBD"))
    q = np.array(q).T
    w = pysal.rook_from_shapefile("examples/columbus.shp")
    w.transform = 'r'
    reg = GM_Error_Het(y, X, w)
    print "Exogenous variables only:"
    print "Dependent variable: CRIME"
    print "Variable  Coef.  S.E."
    print "Constant %5.4f %5.4f" % (reg.betas[0],np.sqrt(reg.vm.diagonal())[0])
    for i in range(len(reg.betas)-2):
        print "Var_%s %5.4f %5.4f" % (i+1,reg.betas[i+1],np.sqrt(reg.vm.diagonal())[i+1])
    print "Lambda: %5.4f %5.4f" % (reg.betas[-1],np.sqrt(reg.vm.diagonal())[-1])
    print '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
    print "Spatial Lag:"
    reg = GM_Combo_Het(y, X, w, yd, q)
    print "Dependent variable: CRIME"
    print "Variable  Coef.  S.E."
    print "Constant %5.4f %5.4f" % (reg.betas[0],np.sqrt(reg.vm.diagonal())[0])
    for i in range(len(reg.betas)-2):
        print "Var_%s %5.4f %5.4f" % (i+1,reg.betas[i+1],np.sqrt(reg.vm.diagonal())[i+1])
    print "Lambda: %5.4f %5.4f" % (reg.betas[-1],np.sqrt(reg.vm.diagonal())[-1])
